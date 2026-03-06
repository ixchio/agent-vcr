window.BENCHMARK_DATA = {
  "lastUpdate": 1772828836081,
  "repoUrl": "https://github.com/ixchio/agent-vcr",
  "entries": {
    "Agent VCR Benchmarks": [
      {
        "commit": {
          "author": {
            "email": "priyankapandeykum@gmail.com",
            "name": "amankumarpandeyin",
            "username": "ixchio"
          },
          "committer": {
            "email": "priyankapandeykum@gmail.com",
            "name": "amankumarpandeyin",
            "username": "ixchio"
          },
          "distinct": true,
          "id": "d4da85280c36dc61afb4c9b77aa9b507451aa4d3",
          "message": "ci: add write permissions for github payload to deploy gh-pages benchmark",
          "timestamp": "2026-03-07T01:50:30+05:30",
          "tree_id": "e3d71d2c0e01e1eb4b9eaba4d160d13f65ece2f7",
          "url": "https://github.com/ixchio/agent-vcr/commit/d4da85280c36dc61afb4c9b77aa9b507451aa4d3"
        },
        "date": 1772828506492,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_performance.py::TestPerformanceBenchmarks::test_benchmark_recorder_overhead",
            "value": 69729.2788639709,
            "unit": "iter/sec",
            "range": "stddev: 0.00001717903235135004",
            "extra": "mean: 14.34117800000223 usec\nrounds: 1000"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestPerformanceBenchmarks::test_benchmark_file_write_speed",
            "value": 3.5927675706193143,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 278.33695900000066 msec\nrounds: 1"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestPerformanceBenchmarks::test_benchmark_load_speed",
            "value": 6.4732810980185835,
            "unit": "iter/sec",
            "range": "stddev: 0.012036275796791925",
            "extra": "mean: 154.48116416666835 msec\nrounds: 6"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestPerformanceBenchmarks::test_benchmark_goto_performance",
            "value": 841327.345216362,
            "unit": "iter/sec",
            "range": "stddev: 5.941377264403885e-7",
            "extra": "mean: 1.1885980001551388 usec\nrounds: 1000"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "priyankapandeykum@gmail.com",
            "name": "amankumarpandeyin",
            "username": "ixchio"
          },
          "committer": {
            "email": "priyankapandeykum@gmail.com",
            "name": "amankumarpandeyin",
            "username": "ixchio"
          },
          "distinct": true,
          "id": "879c3730a7e4575ead99da90dd53a3c44b937440",
          "message": "docs: Update GitHub repository",
          "timestamp": "2026-03-07T01:55:55+05:30",
          "tree_id": "db7d5c474278f6410b090777cc67bc9e7c06cf38",
          "url": "https://github.com/ixchio/agent-vcr/commit/879c3730a7e4575ead99da90dd53a3c44b937440"
        },
        "date": 1772828835678,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/benchmarks/test_performance.py::TestPerformanceBenchmarks::test_benchmark_recorder_overhead",
            "value": 56447.482222438826,
            "unit": "iter/sec",
            "range": "stddev: 0.000014128198924165423",
            "extra": "mean: 17.715581999908636 usec\nrounds: 1000"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestPerformanceBenchmarks::test_benchmark_file_write_speed",
            "value": 3.2697137499671283,
            "unit": "iter/sec",
            "range": "stddev: 0",
            "extra": "mean: 305.83716999998956 msec\nrounds: 1"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestPerformanceBenchmarks::test_benchmark_load_speed",
            "value": 5.908085485872953,
            "unit": "iter/sec",
            "range": "stddev: 0.014235305181730993",
            "extra": "mean: 169.25956850000526 msec\nrounds: 6"
          },
          {
            "name": "tests/benchmarks/test_performance.py::TestPerformanceBenchmarks::test_benchmark_goto_performance",
            "value": 733786.0791422804,
            "unit": "iter/sec",
            "range": "stddev: 6.379049958669018e-7",
            "extra": "mean: 1.362795000375172 usec\nrounds: 1000"
          }
        ]
      }
    ]
  }
}