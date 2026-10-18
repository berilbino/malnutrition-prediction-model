@echo off
title Hadoop MapReduce Execution
echo ======================================================================
echo Executing Apache Hadoop MapReduce on 100,000 Child Survey Records...
echo ======================================================================
cd /d "%~dp0"
python hadoop_mapreduce\run_mapreduce.py
pause
