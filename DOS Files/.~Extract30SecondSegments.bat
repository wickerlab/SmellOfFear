@echo off 
SET /A i = 0 
:loop 

IF %i%==300 GOTO END 
SET /a start=%i%*30 
SET /a end=(%i%+1)*30
set output=Hobbit 2_%i%.mp4
ffmpeg -i "Hobbit2.mp4" -ss %start% -to %end% "%output%"
SET /a i=%i%+1 
GOTO :LOOP 
:END