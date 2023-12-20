conda activate alpa


ssh root@192.168.100.3 -p 9999 "conda activate alpa"
ssh root@192.168.0.4 -p 9999 "conda activate alpa"


ssh root@192.168.100.3 -p 9999 "ray stop"
ray stop
ssh root@192.168.0.4 -p 9999 "ray stop"


ssh root@192.168.100.3 -p 9999 "ray start --head"
ray start --address="192.168.100.3:6379"
ssh root@192.168.0.4 -p 9999 "ray start --address='192.168.100.3:6379'"

ssh root@192.168.0.4 -p 9999 "ray start --head --port=6004"
ray start --address="102.168.0.4:6004"

ssh root@192.168.100.3 -p 9999 "ray start --head --port=6103"