::ProcessFrames 

set /p movie_name=Enter movie file name:
ffmpeg -i TheHungerGames-CatchingFire.mp4 -vf fps=1/3 %movie_name%%04d.jpg -hide_banner