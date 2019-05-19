@echo off 
SET /A i = 199
:loop 

IF %i%==210 GOTO END 
SET /a start=%i%*30 
SET /a end=(%i%+1)*30
set output=Machete Kills_%i%.mp4
ffmpeg -ss %start% -to %end% -i "Machete Kills.mp4" "%output%"
SET /a i=%i%+1 
GOTO :LOOP 
:END