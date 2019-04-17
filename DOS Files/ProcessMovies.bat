::Process Movies
::Purpose: DOS Script that requires the ffmpeg library for audio extraction
::Note: Movie names must not have spaces


@echo off

::movie path
set /p movie_path=Enter path to movie:


::move to path where the movie is located 
cd %movie_path%

::name of movie file with ext
set /p movie_name_ext=Enter movie file name with file ext.:

::name of movie file without ext
set /p movie_name=Enter movie file name without file ext.:

::extract audio from movie 
::convert movie file to wav 
::output is encoded at 192kbps
ffmpeg -i %movie_name_ext% -f wav -ab 192000 -vn %movie_name%.wav 

pause