cd ../isaaclab
git pull --ff-only origin release/3.0.0
rsync -avP skills/ ../space_robotics_bench_l/.agents/skills