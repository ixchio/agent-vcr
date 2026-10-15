# Railway Hosted Control Plane

This guide deploys the optional Agent VCR account and billing control plane.
It does **not** expose local recordings by default: retain the local-first
recorder for private work, and only attach a public product UI after its tenant
storage and authorization layer is ready.

The deployment has four pieces:

- **Railway** runs the Python container and a Railway Postgres service.
- **Supabase Auth** signs user access tokens.
- **Stripe** hosts subscription checkout and sends signed billing events.
- A **Railway Volume** mounted at `/data` holds the existing self-hosted VCR
  project store. Use one replica while it is volume-backed.

## 1. Create the external services

1. In Supabase, create a project and configure your allowed Site URL and auth
   redirect URLs. Use asymmetric JWT signing keys (RS256 or ES256). The server
   verifies the token's signature, issuer, audience, and expiry through the
   public JWKS endpoint; it never needs a Supabase service-role key.
2. In Stripe, create one recurring Price for the initial `pro` plan. Create a
   webhook endpoint at:

   ```text
   https://YOUR_RAILWAY_DOMAIN/api/hosted/billing/webhook
   ```

   Subscribe it to `checkout.session.completed`,
   `customer.subscription.created`, `customer.subscription.updated`, and
   `customer.subscription.deleted`. Copy the endpoint's `whsec_...` signing
   secret, not a webhook test payload.
3. In Railway, create a project with a **PostgreSQL** service and an app
   service connected to this repository. Add a Volume to the app service at
   `/data`. For an existing Railway service that uses Config as Code, set the config file path in
   the app service settings to `/deploy/railway.json`.

## 2. Set Railway variables

Copy the keys in [`deploy/.env.hosted.example`](../../deploy/.env.hosted.example) into the
Railway app's **Variables** panel. Set `DATABASE_URL` as a reference to the
Postgres service's internal `DATABASE_URL`; do not paste a database password
into Git.

Use the generated Railway public domain as `AGENT_VCR_APP_URL` temporarily, or
your final HTTPS application domain once it exists. The value must be an origin
without a path, query, or trailing route.

[`deploy/railway.json`](../../deploy/railway.json) runs the idempotent database migration before
each deployment and requires `/api/health` to return a successful health check before Railway marks
the deployment active.

## 3. Deploy and verify

Railway finds the repository's `Dockerfile` automatically. It installs the
`dashboard` and `hosted` extras and runs:

```text
vcr-server --host 0.0.0.0 --port $PORT --vcr-dir /data --data-dir /data --hosted
```

After Railway reports a successful health check, verify:

```bash
curl -fsS https://YOUR_RAILWAY_DOMAIN/api/health
```

From an authenticated application client, send the Supabase access token:

```bash
curl -fsS https://YOUR_RAILWAY_DOMAIN/api/hosted/me \
  -H "Authorization: Bearer YOUR_SUPABASE_ACCESS_TOKEN"
```

The response initially reports a `free` plan. `POST
/api/hosted/billing/checkout` returns a Stripe-hosted Checkout URL. Do not mark
an account paid from a browser success page: entitlement changes only after a
valid signed Stripe webhook is stored in Postgres. Duplicate webhooks are safe.

## Endpoint contract

| Endpoint | Authentication | Purpose |
| --- | --- | --- |
| `GET /api/hosted/me` | Supabase bearer token | Current user and entitlement state |
| `POST /api/hosted/billing/checkout` | Supabase bearer token | Start a Stripe-hosted Pro subscription checkout |
| `POST /api/hosted/billing/portal` | Supabase bearer token | Open Stripe's customer billing portal |
| `POST /api/hosted/billing/webhook` | Stripe signature | Idempotently update entitlement state |

## Important production boundary

This deployable control plane does not convert the legacy `/api/projects`
self-hosted file store into a public multitenant recordings API. Keep that API
behind an administrator or project token while it remains volume-backed. Before
selling multi-user cloud recording storage, move recordings to tenant-scoped
object storage and enforce ownership for every project/session operation.
