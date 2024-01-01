import ray
# 创建一个默认的客户端
ray.init("ray://192.168.0.5:10001")

# 连接到其他集群
cli2 = ray.init(address="192.168.100.3:6103", allow_multiple=True)

cli1 = ray.init(address="192.168.0.5:6005", allow_multiple=True)

# 数据被放入默认的集群
obj = ray.put("obj")

with cli1:
    obj1 = ray.put("obj1")

with cli2:
    obj2 = ray.put("obj2")

with cli1:
    assert ray.get(obj1) == "obj1"
    try:
        ray.get(obj2)  # 不允许跨集群操作
    except:
        print("Failed to get object which doesn't belong to this cluster")

with cli2:
    assert ray.get(obj2) == "obj2"
    try:
        ray.get(obj1)  # 不允许跨集群操作
    except:
        print("Failed to get object which doesn't belong to this cluster")
assert "obj" == ray.get(obj)
cli1.disconnect()
cli2.disconnect()