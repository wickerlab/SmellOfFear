::ProcessFrames 

ffmpeg -i "The Hunger Games-Catching Fire".mp4 -vf fps=1/3 "The Hunger Games-Catching Fire"_%04d.jpg -hide_banner
