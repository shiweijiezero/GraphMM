#!/bin/bash
for i in {0..10000..1000}
do
  echo "python construct_road_map.py --start $i --end `expr $i + 1000` --flag 0 >> out.log &"
  python construct_road_map.py --start $i --end `expr $i + 1000` --flag 0 >> out.log &
done
python construct_road_map.py --start 0 --end 1000 --flag 1 >> out.log &
