RAY_ROOT="/opt/conda/envs/ray/bin/ray"

ssh root@192.168.0.5 -p 9022 "${RAY_ROOT} stop"
ssh root@192.168.0.4 -p 9022 "${RAY_ROOT} stop"
ssh root@192.168.0.9 -p 9022 "${RAY_ROOT} stop"


ssh root@192.168.0.5 -p 9022 "${RAY_ROOT} start --head"
ssh root@192.168.0.4 -p 9022 "${RAY_ROOT} start --address='192.168.0.5:6379'"
ssh root@192.168.0.9 -p 9022 "${RAY_ROOT} start --address='192.168.0.5:6379'"

ssh root@192.168.0.5 -p 9022 "${RAY_ROOT} status"
