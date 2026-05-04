# Fresh PR Screening Report

This report consolidates all screened tasks under `data/codebase_adaptation/dev/difficulty`.

Rerun policy:
- Later 50-step / 128K reruns override earlier results for tasks that were rerun.
- Earlier passes are preserved for tasks that were never rerun.

## Repo Summary

| Repo | Pass | Fail | Median Successful Steps |
| --- | ---: | ---: | ---: |
| Textualize/rich | 6 | 0 | 7.5 |
| agronholm/apscheduler | 4 | 0 | 23 |
| fsspec/filesystem_spec | 3 | 5 | 10 |
| hynek/structlog | 2 | 1 | 19.5 |
| jazzband/tablib | 9 | 0 | 11 |
| jd/tenacity | 10 | 0 | 13.5 |
| litestar-org/polyfactory | 1 | 4 | 6 |
| pyapp-kit/psygnal | 4 | 1 | 16 |
| simonw/sqlite-utils | 0 | 3 | - |
| vibrantlabsai/ragas | 1 | 3 | 7 |

## Task Results

### Textualize/rich

- Textualize__rich-3985; status=pass; passed_by=gpt-5.4-mini; steps=7; pr=3985; created=2026-02-17
- Textualize__rich-3989; status=pass; passed_by=gpt-5.4-mini; steps=8; pr=3989; created=2026-02-17
- Textualize__rich-3992; status=pass; passed_by=gpt-5.4-mini; steps=7; pr=3992; created=2026-02-17
- Textualize__rich-3995; status=pass; passed_by=gpt-5.4-mini; steps=15; pr=3995; created=2026-02-17
- Textualize__rich-4062; status=pass; passed_by=gpt-5.4; steps=19; pr=4062; created=2026-04-05
- Textualize__rich-4063; status=pass; passed_by=gpt-5.4-mini; steps=7; pr=4063; created=2026-04-05

### agronholm/apscheduler

- agronholm__apscheduler-981; status=pass; passed_by=gpt-5.4; steps=31; pr=981; created=2024-10-28
- agronholm__apscheduler-1060; status=pass; passed_by=gpt-5.4; pr=1060; created=2025-07-01
- agronholm__apscheduler-1062; status=pass; passed_by=gpt-5.4; steps=23; pr=1062; created=2025-07-02
- agronholm__apscheduler-1079; status=pass; passed_by=gpt-5.4-mini; steps=15; pr=1079; created=2025-09-27

### fsspec/filesystem_spec

- fsspec__filesystem_spec-1905; status=fail; pr=1905; created=2025-08-07
- fsspec__filesystem_spec-1923; status=fail; pr=1923; created=2025-09-28
- fsspec__filesystem_spec-1929; status=pass; passed_by=gpt-5.4-mini; steps=10; pr=1929; created=2025-10-07
- fsspec__filesystem_spec-1931; status=pass; passed_by=gpt-5.4; steps=10; pr=1931; created=2025-10-14
- fsspec__filesystem_spec-1933; status=fail; pr=1933; created=2025-10-17
- fsspec__filesystem_spec-1980; status=fail; pr=1980; created=2026-01-24
- fsspec__filesystem_spec-1982; status=fail; pr=1982; created=2026-01-28
- fsspec__filesystem_spec-2010; status=pass; passed_by=gpt-5.4; steps=10; pr=2010; created=2026-04-11

### hynek/structlog

- hynek__structlog-748; status=pass; passed_by=gpt-5.4-mini; steps=8; pr=748; created=2025-09-11
- hynek__structlog-763; status=pass; passed_by=gpt-5.4; steps=31; pr=763; created=2025-10-18
- hynek__structlog-793; status=fail; pr=793; created=2026-03-12

### jazzband/tablib

- jazzband__tablib-534; status=pass; passed_by=gpt-5.4; steps=9; pr=534; created=2023-01-19
- jazzband__tablib-540; status=pass; passed_by=claude-sonnet-4-6; steps=11; pr=540; created=2023-03-01
- jazzband__tablib-547; status=pass; passed_by=gpt-5.4-mini; steps=14; pr=547; created=2023-04-07
- jazzband__tablib-579; status=pass; passed_by=gpt-5.4-mini; steps=6; pr=579; created=2024-02-24
- jazzband__tablib-584; status=pass; passed_by=gpt-5.4-mini; steps=6; pr=584; created=2024-03-21
- jazzband__tablib-594; status=pass; passed_by=claude-sonnet-4-6; steps=27; pr=594; created=2024-07-09
- jazzband__tablib-595; status=pass; passed_by=claude-sonnet-4-6; steps=10; pr=595; created=2024-07-10
- jazzband__tablib-596; status=pass; passed_by=gpt-5.4; steps=16; pr=596; created=2024-07-10
- jazzband__tablib-613; status=pass; passed_by=gpt-5.4-mini; steps=40; pr=613; created=2025-02-19

### jd/tenacity

- jd__tenacity-597; status=pass; passed_by=claude-sonnet-4-6; steps=17; pr=597; created=2026-02-20
- jd__tenacity-603; status=pass; passed_by=claude-sonnet-4-6; steps=13; pr=603; created=2026-02-24
- jd__tenacity-604; status=pass; passed_by=claude-sonnet-4-6; steps=19; pr=604; created=2026-02-24
- jd__tenacity-606; status=pass; passed_by=gpt-5.4-mini; steps=5; pr=606; created=2026-02-24
- jd__tenacity-609; status=pass; passed_by=gpt-5.4-mini; steps=6; pr=609; created=2026-02-24
- jd__tenacity-610; status=pass; passed_by=gpt-5.4-mini; steps=18; pr=610; created=2026-02-24
- jd__tenacity-611; status=pass; passed_by=gpt-5.4; steps=15; pr=611; created=2026-02-24
- jd__tenacity-614; status=pass; passed_by=claude-sonnet-4-6; steps=12; pr=614; created=2026-02-24
- jd__tenacity-615; status=pass; passed_by=gpt-5.4-mini; steps=14; pr=615; created=2026-02-24
- jd__tenacity-628; status=pass; passed_by=claude-sonnet-4-6; steps=10; pr=628; created=2026-03-11

### litestar-org/polyfactory

- litestar-org__polyfactory-711; status=fail; pr=711; created=2025-06-05
- litestar-org__polyfactory-712; status=pass; passed_by=gpt-5.4-mini; steps=6; pr=712; created=2025-06-10
- litestar-org__polyfactory-728; status=fail; pr=728; created=2025-07-05
- litestar-org__polyfactory-770; status=fail; pr=770; created=2025-10-29
- litestar-org__polyfactory-784; status=fail; pr=784; created=2025-11-16

### pyapp-kit/psygnal

- pyapp-kit__psygnal-336; status=pass; passed_by=gpt-5.4-mini; steps=5; pr=336; created=2024-11-08
- pyapp-kit__psygnal-339; status=pass; passed_by=gpt-5.4; pr=339; created=2024-11-15
- pyapp-kit__psygnal-340; status=pass; passed_by=gpt-5.4-mini; steps=42; pr=340; created=2024-11-21
- pyapp-kit__psygnal-346; status=fail; pr=346; created=2024-12-20
- pyapp-kit__psygnal-379; status=pass; passed_by=gpt-5.4; steps=16; pr=379; created=2025-07-01

### simonw/sqlite-utils

- simonw__sqlite-utils-468; status=fail; pr=468; created=2022-08-23
- simonw__sqlite-utils-519; status=fail; pr=519; created=2022-12-21
- simonw__sqlite-utils-526; status=fail; pr=526; created=2023-02-09

### vibrantlabsai/ragas

- vibrantlabsai__ragas-2507; status=pass; passed_by=gpt-5.4-mini; steps=7; pr=2507; created=2025-12-12
- vibrantlabsai__ragas-2545; status=fail; pr=2545; created=2025-12-29
- vibrantlabsai__ragas-2561; status=fail; pr=2561; created=2026-01-14
- vibrantlabsai__ragas-2565; status=fail; pr=2565; created=2026-01-16

