# Deploy on Railway — the hosted control plane 🚀

Run `vcr-server` on Railway with Supabase sign-in, Stripe billing, Postgres for entitlements, and a
volume for the self-hosted project store.

**TL;DR**

- The container listens on `0.0.0.0`, so it **won't start without a token**. Set
  `AGENT_VCR_ADMIN_TOKEN` before your first deploy.
- `deploy/railway.json` runs the database migration before every deploy and waits for
  `/api/health` before switching traffic.
- This deploys the account and billing layer. Recordings stay local-first; the project store sits
  behind admin/project tokens and is **not** a multi-tenant recordings API.

## Try it in 60 seconds (locally)

Before touching Railway, see the guard rail that protects you there:

```bash
pip install "ai-agent-vcr[dashboard]"
vcr serve --host 0.0.0.0
# error: Refusing to serve the Agent VCR API on non-loopback host '0.0.0.0' without
# authentication. Set AGENT_VCR_AUTH_TOKEN and/or AGENT_VCR_ADMIN_TOKEN ...
```

The server exits with code 2 and never opens the port. Railway runs the same check, so a deploy
without a token fails fast instead of exposing an open API.

## How it works

```text
           browser / app                        Railway project
  ┌───────────────────────────┐     ┌──────────────────────────────────────────┐
  │ Supabase access token ────┼────▶│ app service: vcr-server --hosted         │
  │ (RS256/ES256 JWT)         │     │   /api/hosted/*  ── Supabase JWKS check  │
  └───────────────────────────┘     │   /api/projects/* ── admin/project token │
  Stripe ── signed webhook ────────▶│   Volume /data (project store, 1 replica)│
                                    │        │                                 │
                                    │        ▼                                 │
                                    │ PostgreSQL service (accounts, billing)   │
                                    └──────────────────────────────────────────┘
```

- **Railway** runs the Python container (from the repo's `Dockerfile`) and a Postgres service.
- **Supabase Auth** signs user access tokens. The server checks signature, issuer, audience and
  expiry against Supabase's public JWKS endpoint. It never needs a service-role key.
- **Stripe** hosts subscription checkout and sends signed billing events.
- A **Railway Volume** at `/data` holds the self-hosted VCR project store. Run **one replica**
  while it is volume-backed.

## Step 1 — Create the external services

1. **Supabase:** create a project. Configure the allowed Site URL and auth redirect URLs. Use
   asymmetric JWT signing keys (RS256 or ES256).
2. **Stripe:** create one recurring Price for the initial `pro` plan. Add a webhook endpoint at:

   ```text
   https://YOUR_RAILWAY_DOMAIN/api/hosted/billing/webhook
   ```

   Subscribe it to `checkout.session.completed`, `customer.subscription.created`,
   `customer.subscription.updated` and `customer.subscription.deleted`. Copy the endpoint's
   `whsec_...` signing secret (not a test payload).
3. **Railway:** create a project with a **PostgreSQL** service and an app service connected to this
   repository. Add a Volume to the app service at `/data`. If the service uses Config as Code, set
   the config file path in the app service settings to `/deploy/railway.json`.

## Step 2 — Set Railway variables

Copy the keys from [`deploy/.env.hosted.example`](../../deploy/.env.hosted.example) into the app's
**Variables** panel:

| Variable | Required | Notes |
| --- | --- | --- |
| `DATABASE_URL` | Yes | Reference the Postgres service's internal URL (`${{Postgres.DATABASE_URL}}`). Never paste a password into Git. Must be `postgres://` or `postgresql://`. |
| `AGENT_VCR_SUPABASE_URL` | Yes | `https://YOUR_PROJECT.supabase.co` (HTTPS origin, no path) |
| `AGENT_VCR_SUPABASE_AUDIENCE` | No | Defaults to `authenticated` |
| `AGENT_VCR_APP_URL` | Yes | Your app's origin, also used as the CORS origin. Start with the Railway public domain, switch to your final HTTPS domain later. Origin only: no path, query or trailing route. |
| `AGENT_VCR_STRIPE_SECRET_KEY`, `AGENT_VCR_STRIPE_WEBHOOK_SECRET`, `AGENT_VCR_STRIPE_PRICE_ID` | All three, or none | Billing is enabled only when all three are set; a partial set fails at startup. |

`PORT` is provided by Railway automatically.

### Authentication variables 🔐

Generate each secret with `python -c "import secrets; print(secrets.token_urlsafe(32))"` and keep
it only in Railway Variables:

| Variable | Required | Purpose |
| --- | --- | --- |
| `AGENT_VCR_ADMIN_TOKEN` | Yes (recommended) | Admin token: create projects (`POST /api/projects`), mint project tokens (`POST /api/projects/{id}/tokens`), and everything else. |
| `AGENT_VCR_AUTH_TOKEN` | No | A second, separate secret with the **same admin rights** — handy for the dashboard API, recorders (`/api/push`) and the `/ws/live` WebSocket. |
| `AGENT_VCR_ALLOW_UNAUTHENTICATED` | No | `1` starts with no token on a public host. Insecure; never set it in production. |

Strictly, the server needs **at least one** of the two tokens to start; set the admin token so
there's always a clear owner. These variables are the defaults for the `--admin-token`,
`--auth-token` and `--allow-unauthenticated` flags, and an explicit flag wins. Prefer the
variables in containers, so secrets never show up in the process list.

With a token configured:

- every `/api` route except `/api/health` and `/api/hosted/*` needs
  `Authorization: Bearer <token>` (or `x-agent-vcr-token: <token>`);
- project tokens work only on that project's `/api/projects/{id}/...` routes;
- `/ws/live` also accepts `?token=<token>`, because browsers can't set WebSocket headers. HTTP
  routes don't.

Details on tokens and the project store: [../dashboard.md](../dashboard.md).

## Step 3 — Deploy and verify ✅

Railway finds the repository's `Dockerfile` automatically. It installs the `dashboard` and `hosted`
extras, runs as a non-root user, and starts:

```text
vcr-server --host 0.0.0.0 --port $PORT --vcr-dir /data --data-dir /data \
  --allowed-origin $AGENT_VCR_APP_URL --hosted
```

[`deploy/railway.json`](../../deploy/railway.json) adds:

- `python -m agent_vcr.hosted_migrate` as the pre-deploy command (idempotent migration);
- a health check on `/api/health` (120 s timeout) before the deployment goes active;
- restart on failure (up to 3 retries) and 20 s connection draining.

Once Railway reports a healthy deploy:

```bash
curl -fsS https://YOUR_RAILWAY_DOMAIN/api/health
# {"status":"ok","version":"1.7.0","auth":true}

# The self-hosted API must reject anonymous calls (expect 401):
curl -s -o /dev/null -w '%{http_code}\n' https://YOUR_RAILWAY_DOMAIN/api/sessions
```

From a signed-in application client, send the Supabase access token:

```bash
curl -fsS https://YOUR_RAILWAY_DOMAIN/api/hosted/me \
  -H "Authorization: Bearer YOUR_SUPABASE_ACCESS_TOKEN"
```

A new account starts on the `free` plan. `POST /api/hosted/billing/checkout` returns a
Stripe-hosted Checkout URL. Entitlements change **only** after a valid, signed Stripe webhook is
stored in Postgres — never because a browser reached a success page. Duplicate webhooks are safe.

## What it can't do

- **It's not multi-tenant recording storage.** The `/api/projects` store is files on one volume.
  Keep it behind an admin or project token. Before selling multi-user cloud recording storage,
  move recordings to tenant-scoped object storage and enforce ownership on every project/session
  operation.
- **One replica only** while the store lives on a Railway Volume.
- **No local recordings are exposed by default.** Keep recording locally for private work; attach
  a public product UI only once its tenant storage and authorization layer is ready.
- **The bundled web dashboard can't send tokens yet**, so on a token-protected deployment use the
  API and `vcr cloud` ([../dashboard.md](../dashboard.md)).

## Cheat sheet

| Endpoint | Authentication | Purpose |
| --- | --- | --- |
| `GET /api/health` | none | Health check (`auth: true` when a token is set) |
| `GET /api/hosted/me` | Supabase bearer token | Current user and entitlement state |
| `POST /api/hosted/billing/checkout` | Supabase bearer token | Start a Stripe-hosted Pro checkout |
| `POST /api/hosted/billing/portal` | Supabase bearer token | Open Stripe's customer billing portal |
| `POST /api/hosted/billing/webhook` | Stripe signature | Idempotently update entitlements |
| `/api/projects/...`, `/api/sessions/...`, `/ws/live` | Admin or project token | Self-hosted store and dashboard API |

## Next

- [../dashboard.md](../dashboard.md) — tokens, the project store and `vcr cloud`.
- [../integrations/n8n.md](../integrations/n8n.md) — trigger workflows on your hosted server.
- [../README.md](../README.md) — all guides.
