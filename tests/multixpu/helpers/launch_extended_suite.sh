#!/usr/bin/env bash
# Launch the extended 2-GPU suite + automatic root-cause analysis, fully detached.
# In its own file so the caller's command line cannot collide with the suite's
# internal pkill patterns.
cd /home/sdp/madhu/OpenRLHF-fresh
exec /home/sdp/venvs/openrlhf-xccl-auto-detect-213/bin/python \
     tests/run_multigpu_extended_with_rca.py
