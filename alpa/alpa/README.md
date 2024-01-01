# 文件介绍

+ AAA_main_file 文件夹里面的都是用来作为测试的入口函数， 
    + 1_maintest.py 实现了双层mlp的model 并用benchmark来启动
    + nccl.py 是做通信测试


+ util.py 增加了 print_jaxpr_computation_graph 函数用于debug, 功能是以容易阅读的方式打印出jaxpr
+ device_mesh 增加了关于ray的启动print, 用于debug
+ api.py 用来在 alpa.grad 上打断点单步执行
+ pipeline_parallel.layer... 增添了dp对应的注释, 详细参考 https://spicy-scribe-981.notion.site/Layer_construction-e0efe629d6aa4e50bc865d135b5304d4?pvs=4