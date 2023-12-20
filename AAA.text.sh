ray start --head --resources='{"cluster_H100": 1}'
ray start --head --resources='{"cluster_A100": 1}'
ray start --address='192.168.0.4:6379' --resources='{"cluster_A100": 1}'
ray start --address='192.168.0.4:6379' --resources='{"cluster_H100": 1}'