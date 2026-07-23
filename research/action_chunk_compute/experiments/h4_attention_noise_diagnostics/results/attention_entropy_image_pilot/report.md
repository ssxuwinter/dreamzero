# H5 attention entropy image pilot

- episode：3
- chunk：3
- 数据：`image/` 下三个 episode 的三视角 mp4。

## Summary

|   episode |   chunk_index |   anchor | stage     |   latency |   record_count |   mean_entropy |   std_entropy |   min_entropy |   max_entropy |
|----------:|--------------:|---------:|:----------|----------:|---------------:|---------------:|--------------:|--------------:|--------------:|
|      1925 |             0 |       23 | free_open |   34.4215 |            680 |       0.621849 |     0.0579435 |      0.484508 |      0.786413 |
|     16001 |             0 |       23 | free_open |   21.4572 |            680 |       0.618295 |     0.0586478 |      0.477272 |      0.78717  |
|     50458 |             0 |       23 | free_open |   21.4799 |            680 |       0.620827 |     0.0581826 |      0.482596 |      0.785222 |

## By stage

| stage     |   chunks |   mean_entropy |   std_entropy |   mean_records |
|:----------|---------:|---------------:|--------------:|---------------:|
| free_open |        3 |       0.620324 |    0.00182933 |            680 |

## Pilot interpretation

该 pilot 只验证 attention entropy 插桩与数据通路是否可用，样本数不足以支持 compute gate 结论。若 record_count 为 0，说明模型路径未触发当前插桩标签或服务端未返回诊断字段。
