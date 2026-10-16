| arm | 질의 | 사례 | 단계 | 모드 | 응답(ms) | 서버 rerank(ms) · shape | 가사 경로(ms) | 상태 | stage | calls/applied | 가짜 호출[결과] | 정답 순위(후보 top10 순위) | 결과 수/중복/후보 안 | top-3 | v55 리랭킹 전 top-10 일치 | 오류 |
|---|---|---|---|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| A | t014 | repro_v55 | 1 | normal (=1st normal) | 508.0 | 1.9 plain | 13.7 | 200 | model_applied | 1/1 | 1[returned] | -(1) | 15/0/True | 36011504|34187530|602677292 | no |  |
| A | t014 | repro_v55 | 2 | exception | 2156.6 | 2024.5 plain | 13.0 | 200 | model_failed | 1/0 | 3[raised|raised|raised] | 1(1) | 15/0/True | 609293|106465|525682 | equal |  |
| A | t014 | repro_v55 | 3 | slow | 31208.2 | 31015.7 plain | 25.1 | 200 | model_failed | 1/0 | 2[cancelled|cancelled] | 1(1) | 15/0/True | 609293|106465|525682 | equal |  |
| A | t014 | repro_v55 | 4 | invalid_json | 2238.6 | 2024.2 plain | 26.8 | 200 | model_failed | 1/0 | 3[returned|returned|returned] | 1(1) | 15/0/True | 609293|106465|525682 | equal |  |
| A | t014 | repro_v55 | 5 | foreign_ids | 182.5 | 1.5 plain | 31.6 | 200 | model_applied | 1/1 | 1[returned] | 1(1) | 15/0/True | 609293|106465|525682 | equal |  |
| A | t014 | repro_v55 | 6 | slow_within | 8304.3 | 8004.7 plain | 12.0 | 200 | model_applied | 1/1 | 1[returned] | -(1) | 15/0/True | 36011504|34187530|602677292 | no |  |
| A | t014 | repro_v55 | 7 | normal (=1st normal) | 184.1 | 1.5 plain | 28.0 | 200 | model_applied | 1/1 | 1[returned] | -(1) | 15/0/True | 36011504|34187530|602677292 | no |  |
| A | t049 | repro_v55 | 1 | normal (=1st normal) | 155.2 | 1.5 plain |  | 200 | model_applied | 1/1 | 1[returned] | -(>10) | 15/0/True | 8252409|4205993|602389135 |  |  |
| A | t049 | repro_v55 | 2 | exception | 2150.6 | 2026.0 plain |  | 200 | model_failed | 1/0 | 3[raised|raised|raised] | 12(>10) | 15/0/True | 1871159|4832150|4004361 |  |  |
| A | t049 | repro_v55 | 3 | slow | 31227.5 | 31011.8 plain |  | 200 | model_failed | 1/0 | 2[cancelled|cancelled] | 12(>10) | 15/0/True | 1871159|4832150|4004361 |  |  |
| A | t049 | repro_v55 | 4 | invalid_json | 2233.8 | 2020.7 plain |  | 200 | model_failed | 1/0 | 3[returned|returned|returned] | 12(>10) | 15/0/True | 1871159|4832150|4004361 |  |  |
| A | t049 | repro_v55 | 5 | foreign_ids | 173.1 | 1.5 plain |  | 200 | model_applied | 1/1 | 1[returned] | 12(>10) | 15/0/True | 1871159|4832150|4004361 |  |  |
| A | t049 | repro_v55 | 6 | slow_within | 8125.3 | 8006.5 plain |  | 200 | model_applied | 1/1 | 1[returned] | -(>10) | 15/0/True | 8252409|4205993|602389135 |  |  |
| A | t049 | repro_v55 | 7 | normal (=1st normal) | 182.2 | 1.5 plain |  | 200 | model_applied | 1/1 | 1[returned] | -(>10) | 15/0/True | 8252409|4205993|602389135 |  |  |
| A | n007 | lyric_protected | 1 | normal (=1st normal) | 166.0 | 2.3 protected | 9.7 | 200 | model_applied | 3/3 | 3[returned|returned|returned] | - | 15/0/True | 460396|33691794|30588627 |  |  |
| A | n007 | lyric_protected | 2 | exception | 6365.8 | 6052.0 protected | 6.4 | 200 | model_failed | 3/0 | 9[raised|raised|raised|raised|raised|raised|raised|raised|raised] | - | 15/0/True | 460396|33691794|30588627 |  |  |
| A | n007 | lyric_protected | 3 | fail_group_first | 2226.4 | 2021.2 protected | 17.6 | 200 | model_applied | 3/2 | 5[raised|raised|raised|returned|returned] | - | 15/0/True | 460396|33691794|30588627 |  |  |
| A | n007 | lyric_protected | 4 | slow | 93252.3 | 93034.7 protected | 21.3 | 200 | model_failed | 3/0 | 6[cancelled|cancelled|cancelled|cancelled|cancelled|cancelled] | - | 15/0/True | 460396|33691794|30588627 |  |  |
| A | n007 | lyric_protected | 5 | normal (=1st normal) | 350614.7 | 3.4 protected | 350591.0 | 200 | model_applied | 3/3 | 3[returned|returned|returned] | - | 15/0/True | 460396|33691794|30588627 |  |  |
| A | t018 | lyric_protected | 1 | normal (=1st normal) | 600.0 | 2.2 protected | 20.8 | 200 | model_applied | 2/2 | 2[returned|returned] | - | 15/0/True | 21360|34997081|602540011 |  |  |
| A | t018 | lyric_protected | 2 | exception | 4166.1 | 4044.6 protected | 15.5 | 200 | model_failed | 2/0 | 6[raised|raised|raised|raised|raised|raised] | - | 15/0/True | 21360|34997081|4123405 |  |  |
| A | t018 | lyric_protected | 3 | fail_group_first | 2212.1 | 2022.2 protected | 34.1 | 200 | model_applied | 2/1 | 4[raised|raised|raised|returned] | - | 15/0/True | 21360|34997081|602540011 |  |  |
| A | t018 | lyric_protected | 4 | slow | 62228.1 | 62025.6 protected | 35.9 | 200 | model_failed | 2/0 | 4[cancelled|cancelled|cancelled|cancelled] | - | 15/0/True | 21360|34997081|4123405 |  |  |
| A | t018 | lyric_protected | 5 | normal (=1st normal) | 425.0 | 1.9 protected | 42.1 | 200 | model_applied | 2/2 | 2[returned|returned] | - | 15/0/True | 21360|34997081|602540011 |  |  |
| A | q117 | policy_skip | 1 | exception | 130.7 |   |  | 200 | policy_skipped | 0/0 | 0[] | - | 15/0/True | 521736|521724|521747 |  |  |
| A | q117 | policy_skip | 2 | normal (=1st normal) | 80.4 |   |  | 200 | policy_skipped | 0/0 | 0[] | - | 15/0/True | 521736|521724|521747 |  |  |
| A | q112 | turn2_gender | 1 | normal (=1st normal) | 304.6 | 1.5 plain |  | 200 | model_applied | 1/1 | 1[returned] | - | 15/0/True | 2129151|1929329|2289852 |  |  |
| A | q112 | turn2_gender | 2 | exception | 2159.3 | 2018.4 plain |  | 200 | model_failed | 1/0 | 3[raised|raised|raised] | - | 15/0/True | 1765767|1786851|933372 |  |  |
| A | q112 | turn2_gender | 3 | slow | 31246.6 | 31013.5 plain |  | 200 | model_failed | 1/0 | 2[cancelled|cancelled] | - | 15/0/True | 1765767|1786851|933372 |  |  |
| A | q112 | turn2_gender | 4 | normal (=1st normal) | 234.8 | 1.6 plain |  | 200 | model_applied | 1/1 | 1[returned] | - | 15/0/True | 2129151|1929329|2289852 |  |  |
| A | n011 | turn2_genre | 1 | normal (=1st normal) | 102.1 | 2.3 plain |  | 200 | model_applied | 1/1 | 1[returned] | - | 15/0/True | 7860122|1199786|7854179 |  |  |
| A | n011 | turn2_genre | 2 | exception | 2138.6 | 2017.2 plain |  | 200 | model_failed | 1/0 | 3[raised|raised|raised] | - | 15/0/True | 1875215|1322134|2648642 |  |  |
| A | n011 | turn2_genre | 3 | slow | 31215.2 | 31015.7 plain |  | 200 | model_failed | 1/0 | 2[cancelled|cancelled] | - | 15/0/True | 1875215|1322134|2648642 |  |  |
| A | n011 | turn2_genre | 4 | normal (=1st normal) | 167.5 | 1.8 plain |  | 200 | model_applied | 1/1 | 1[returned] | - | 15/0/True | 7860122|1199786|7854179 |  |  |
| D | t014 | repro_v55 | 1 | normal (=1st normal) | 351277.4 | 3.4 plain | 351249.9 | 200 | model_applied | 1/1 | 1[returned] | -(10) | 15/0/True | 1839831|495147|504315 | no |  |
| D | t014 | repro_v55 | 2 | exception | 2626.4 | 2023.2 plain | 11.7 | 200 | model_failed | 1/0 | 3[raised|raised|raised] | 10(10) | 15/0/True | 106465|621087|24670 | equal |  |
| D | t014 | repro_v55 | 3 | slow | 31258.8 | 31013.4 plain | 27.1 | 200 | model_failed | 1/0 | 2[cancelled|cancelled] | 10(10) | 15/0/True | 106465|621087|24670 | equal |  |
| D | t014 | repro_v55 | 4 | invalid_json | 2568.5 | 2020.7 plain | 37.2 | 200 | model_failed | 1/0 | 3[returned|returned|returned] | 10(10) | 15/0/True | 106465|621087|24670 | equal |  |
| D | t014 | repro_v55 | 5 | foreign_ids | 453.0 | 1.6 plain | 27.5 | 200 | model_applied | 1/1 | 1[returned] | 10(10) | 15/0/True | 106465|621087|24670 | equal |  |
| D | t014 | repro_v55 | 6 | slow_within | 8174.8 | 8006.6 plain | 14.2 | 200 | model_applied | 1/1 | 1[returned] | -(10) | 15/0/True | 1839831|495147|504315 | no |  |
| D | t014 | repro_v55 | 7 | normal (=1st normal) | 234.6 | 1.7 plain | 29.6 | 200 | model_applied | 1/1 | 1[returned] | -(10) | 15/0/True | 1839831|495147|504315 | no |  |
| D | t049 | repro_v55 | 1 | normal (=1st normal) | 197.5 | 1.6 plain |  | 200 | model_applied | 1/1 | 1[returned] | 13(>10) | 15/0/True | 602377142|36355357|31684632 |  |  |
| D | t049 | repro_v55 | 2 | exception | 2172.0 | 2022.3 plain |  | 200 | model_failed | 1/0 | 3[raised|raised|raised] | -(>10) | 15/0/True | 3110565|3534943|1871159 |  |  |
| D | t049 | repro_v55 | 3 | slow | 31238.4 | 31006.0 plain |  | 200 | model_failed | 1/0 | 2[cancelled|cancelled] | -(>10) | 15/0/True | 3110565|3534943|1871159 |  |  |
| D | t049 | repro_v55 | 4 | invalid_json | 2255.7 | 2014.2 plain |  | 200 | model_failed | 1/0 | 3[returned|returned|returned] | -(>10) | 15/0/True | 3110565|3534943|1871159 |  |  |
| D | t049 | repro_v55 | 5 | foreign_ids | 201.5 | 1.6 plain |  | 200 | model_applied | 1/1 | 1[returned] | -(>10) | 15/0/True | 3110565|3534943|1871159 |  |  |
| D | t049 | repro_v55 | 6 | slow_within | 8151.4 | 8005.0 plain |  | 200 | model_applied | 1/1 | 1[returned] | 13(>10) | 15/0/True | 602377142|36355357|31684632 |  |  |
| D | t049 | repro_v55 | 7 | normal (=1st normal) | 198.7 | 1.5 plain |  | 200 | model_applied | 1/1 | 1[returned] | 13(>10) | 15/0/True | 602377142|36355357|31684632 |  |  |
| D | n007 | lyric_protected | 1 | normal (=1st normal) | 165.9 | 2.3 protected | 9.6 | 200 | model_applied | 3/3 | 3[returned|returned|returned] | - | 15/0/True | 460396|33691794|30588627 |  |  |
| D | n007 | lyric_protected | 2 | exception | 6375.1 | 6065.8 protected | 6.5 | 200 | model_failed | 3/0 | 9[raised|raised|raised|raised|raised|raised|raised|raised|raised] | - | 15/0/True | 460396|33691794|30588627 |  |  |
| D | n007 | lyric_protected | 3 | fail_group_first | 2221.0 | 2025.5 protected | 14.8 | 200 | model_applied | 3/2 | 5[raised|raised|raised|returned|returned] | - | 15/0/True | 460396|33691794|30588627 |  |  |
| D | n007 | lyric_protected | 4 | slow | 93265.6 | 93026.0 protected | 18.0 | 200 | model_failed | 3/0 | 6[cancelled|cancelled|cancelled|cancelled|cancelled|cancelled] | - | 15/0/True | 460396|33691794|30588627 |  |  |
| D | n007 | lyric_protected | 5 | normal (=1st normal) | 199.7 | 2.2 protected | 19.7 | 200 | model_applied | 3/3 | 3[returned|returned|returned] | - | 15/0/True | 460396|33691794|30588627 |  |  |
| D | t018 | lyric_protected | 1 | normal (=1st normal) | 165.6 | 1.9 protected | 19.7 | 200 | model_applied | 2/2 | 2[returned|returned] | - | 15/0/True | 21360|34997081|431368 |  |  |
| D | t018 | lyric_protected | 2 | exception | 4171.4 | 4032.0 protected | 16.0 | 200 | model_failed | 2/0 | 6[raised|raised|raised|raised|raised|raised] | - | 15/0/True | 21360|34997081|37100 |  |  |
| D | t018 | lyric_protected | 3 | fail_group_first | 2242.5 | 2028.9 protected | 35.0 | 200 | model_applied | 2/1 | 4[raised|raised|raised|returned] | - | 15/0/True | 21360|34997081|431368 |  |  |
| D | t018 | lyric_protected | 4 | slow | 62249.2 | 62025.0 protected | 34.8 | 200 | model_failed | 2/0 | 4[cancelled|cancelled|cancelled|cancelled] | - | 15/0/True | 21360|34997081|37100 |  |  |
| D | t018 | lyric_protected | 5 | normal (=1st normal) | 209.8 | 2.0 protected | 33.1 | 200 | model_applied | 2/2 | 2[returned|returned] | - | 15/0/True | 21360|34997081|431368 |  |  |
| D | q117 | policy_skip | 1 | exception | 125.6 |   |  | 200 | policy_skipped | 0/0 | 0[] | - | 15/0/True | 521736|521724|521747 |  |  |
| D | q117 | policy_skip | 2 | normal (=1st normal) | 78.9 |   |  | 200 | policy_skipped | 0/0 | 0[] | - | 15/0/True | 521736|521724|521747 |  |  |
| D | q112 | turn2_gender | 1 | normal (=1st normal) | 318.4 | 1.5 plain |  | 200 | model_applied | 1/1 | 1[returned] | - | 15/0/True | 2803451|862035|2203097 |  |  |
| D | q112 | turn2_gender | 2 | exception | 2189.3 | 2021.8 plain |  | 200 | model_failed | 1/0 | 3[raised|raised|raised] | - | 15/0/True | 1765767|1786851|933372 |  |  |
| D | q112 | turn2_gender | 3 | slow | 31277.4 | 31013.3 plain |  | 200 | model_failed | 1/0 | 2[cancelled|cancelled] | - | 15/0/True | 1765767|1786851|933372 |  |  |
| D | q112 | turn2_gender | 4 | normal (=1st normal) | 262.0 | 2.0 plain |  | 200 | model_applied | 1/1 | 1[returned] | - | 15/0/True | 2803451|862035|2203097 |  |  |
| D | n011 | turn2_genre | 1 | normal (=1st normal) | 126.4 | 2.1 plain |  | 200 | model_applied | 1/1 | 1[returned] | - | 15/0/True | 2099506|1592906|5470838 |  |  |
| D | n011 | turn2_genre | 2 | exception | 2151.2 | 2017.1 plain |  | 200 | model_failed | 1/0 | 3[raised|raised|raised] | - | 15/0/True | 1875215|1951553|1951550 |  |  |
| D | n011 | turn2_genre | 3 | slow | 31206.9 | 31008.7 plain |  | 200 | model_failed | 1/0 | 2[cancelled|cancelled] | - | 15/0/True | 1875215|1951553|1951550 |  |  |
| D | n011 | turn2_genre | 4 | normal (=1st normal) | 223.2 | 1.8 plain |  | 200 | model_applied | 1/1 | 1[returned] | - | 15/0/True | 2099506|1592906|5470838 |  |  |

## 폴백 반환 = 후보 순서(candidate_ids 앞 top_k)인가
- A t014 exception: 결과 top-10 == 후보 top-10 → True (stage=model_failed, calls=1/0)
- A t014 slow: 결과 top-10 == 후보 top-10 → True (stage=model_failed, calls=1/0)
- A t014 invalid_json: 결과 top-10 == 후보 top-10 → True (stage=model_failed, calls=1/0)
- A t014 foreign_ids: 결과 top-10 == 후보 top-10 → True (stage=model_applied, calls=1/1)
- A t049 exception: 결과 top-10 == 후보 top-10 → True (stage=model_failed, calls=1/0)
- A t049 slow: 결과 top-10 == 후보 top-10 → True (stage=model_failed, calls=1/0)
- A t049 invalid_json: 결과 top-10 == 후보 top-10 → True (stage=model_failed, calls=1/0)
- A t049 foreign_ids: 결과 top-10 == 후보 top-10 → True (stage=model_applied, calls=1/1)
- A n007 exception: 결과 top-10 == 후보 top-10 → False (stage=model_failed, calls=3/0)
- A n007 fail_group_first: 결과 top-10 == 후보 top-10 → False (stage=model_applied, calls=3/2)
- A n007 slow: 결과 top-10 == 후보 top-10 → False (stage=model_failed, calls=3/0)
- A t018 exception: 결과 top-10 == 후보 top-10 → True (stage=model_failed, calls=2/0)
- A t018 fail_group_first: 결과 top-10 == 후보 top-10 → False (stage=model_applied, calls=2/1)
- A t018 slow: 결과 top-10 == 후보 top-10 → True (stage=model_failed, calls=2/0)
- A q117 exception: 결과 top-10 == 후보 top-10 → True (stage=policy_skipped, calls=0/0)
- A q112 exception: 결과 top-10 == 후보 top-10 → True (stage=model_failed, calls=1/0)
- A q112 slow: 결과 top-10 == 후보 top-10 → True (stage=model_failed, calls=1/0)
- A n011 exception: 결과 top-10 == 후보 top-10 → True (stage=model_failed, calls=1/0)
- A n011 slow: 결과 top-10 == 후보 top-10 → True (stage=model_failed, calls=1/0)
- D t014 exception: 결과 top-10 == 후보 top-10 → True (stage=model_failed, calls=1/0)
- D t014 slow: 결과 top-10 == 후보 top-10 → True (stage=model_failed, calls=1/0)
- D t014 invalid_json: 결과 top-10 == 후보 top-10 → True (stage=model_failed, calls=1/0)
- D t014 foreign_ids: 결과 top-10 == 후보 top-10 → True (stage=model_applied, calls=1/1)
- D t049 exception: 결과 top-10 == 후보 top-10 → True (stage=model_failed, calls=1/0)
- D t049 slow: 결과 top-10 == 후보 top-10 → True (stage=model_failed, calls=1/0)
- D t049 invalid_json: 결과 top-10 == 후보 top-10 → True (stage=model_failed, calls=1/0)
- D t049 foreign_ids: 결과 top-10 == 후보 top-10 → True (stage=model_applied, calls=1/1)
- D n007 exception: 결과 top-10 == 후보 top-10 → False (stage=model_failed, calls=3/0)
- D n007 fail_group_first: 결과 top-10 == 후보 top-10 → False (stage=model_applied, calls=3/2)
- D n007 slow: 결과 top-10 == 후보 top-10 → False (stage=model_failed, calls=3/0)
- D t018 exception: 결과 top-10 == 후보 top-10 → True (stage=model_failed, calls=2/0)
- D t018 fail_group_first: 결과 top-10 == 후보 top-10 → False (stage=model_applied, calls=2/1)
- D t018 slow: 결과 top-10 == 후보 top-10 → True (stage=model_failed, calls=2/0)
- D q117 exception: 결과 top-10 == 후보 top-10 → True (stage=policy_skipped, calls=0/0)
- D q112 exception: 결과 top-10 == 후보 top-10 → True (stage=model_failed, calls=1/0)
- D q112 slow: 결과 top-10 == 후보 top-10 → True (stage=model_failed, calls=1/0)
- D n011 exception: 결과 top-10 == 후보 top-10 → True (stage=model_failed, calls=1/0)
- D n011 slow: 결과 top-10 == 후보 top-10 → True (stage=model_failed, calls=1/0)
