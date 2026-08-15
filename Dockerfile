# =====================================================================
# 飞轮组件 — 统一复现镜像 (Python 3.12 + torch 2.11.0, 与相控阵同源)
#
#   与相控阵镜像的差异:
#     1. 数据 (~1.65 GB) 不烘焙进镜像, 通过 volume 只读挂载到 /app/release/data
#        (依据: wheel_docker_unified_environment_agent_goal.md 任务1)
#     2. 工作目录 /app/release (飞轮原包结构, scripts/handoff 校验脚本假设此布局)
#     3. 入口脚本放 /app/ 下, 不进 release/, 以免干扰 verify_manifest 470/470 校验
#
#   构建时验证: 关键模块 import 检查 (不依赖 data)
#   运行时验证: docker compose run --rm wheel verify
# =====================================================================
FROM python:3.12-slim

ENV TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=42 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app/release

# 1) 先装 torch 2.11.0 (cu130 源, 与相控阵 Dockerfile 完全一致)
COPY requirements.unified.lock /tmp/requirements.unified.lock
RUN pip install --no-cache-dir --default-timeout=300 --retries=5 torch==2.11.0 \
      --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
      --extra-index-url https://download.pytorch.org/whl/cu130 \
      --trusted-host pypi.tuna.tsinghua.edu.cn \
      --trusted-host download.pytorch.org \
 && pip install --no-cache-dir --default-timeout=300 --retries=5 -r /tmp/requirements.unified.lock \
      --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
      --trusted-host pypi.tuna.tsinghua.edu.cn

# 2) 非 root 用户
RUN useradd -m -u 1000 appuser

# 3) 拷贝飞轮原包 (不含 data/, data 通过 volume 只读挂载)
COPY release/ /app/release/

# 4) 入口脚本 (放 /app/ 下, 不进 release/, 不干扰 manifest 校验)
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

RUN chown -R appuser:appuser /app
USER appuser

ENV PYTHONPATH=/app/release

# 5) 构建时验证: 关键模块 import 检查 (不依赖 data, data 运行时挂载)
RUN python -c "\
import src; \
import src.models.tcn_encoder; \
import src.transfer.train_transfer; \
import src.transfer.mmd; \
import src.train.pretrain; \
import src.sim.wheel_sim; \
import src.sim.build_hi; \
import src.baselines.physical_extrap; \
import src.baselines.wiener_pf; \
print('✅ 飞轮关键模块导入成功')"

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["verify"]
