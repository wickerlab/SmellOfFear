import pickle
from google.cloud import translate

#load subtitle to be translated
subtitlePath = 'C:\\Users\\hasna\\Desktop\\Smell of Fear\\BuddyUntranslated.p'
subtitle = pickle.load(open(subtitlePath, "rb" ))

translatedSubtitles = list()
translate_client = translate.Client()
for section in subtitle:
    # Translates some text from german to english
    if len(section) != 0:
        translation = translate_client.translate(section, target_language='en')
        print(translation)
        translatedSubtitles.append(translation)
    else:
        translatedSubtitles.append([])

#remove all unecessary characters from the translated text
#loop through each segement and each dialog from within each segment
engTranslated = list()
for i in range(0, len(translatedSubtitles)):
    if len(translatedSubtitles[i]) != 0:
        dialog=translatedSubtitles[i]['translatedText']
        #parse the dialog for weird html characters
        k = 0
        while True:
            try:
                if dialog[k] == '&':
                    index = dialog.index(';')
                    #split off special ascii character
                    asciiChr = dialog[k:index+1]
                    if asciiChr == '&#39;':
                        dialog = dialog[:k] + '\'' + dialog[index+1:]
                    elif asciiChr == '&quot;':
                        dialog = dialog[:k] + '\"' + dialog[index+1:]
                k = k + 1
            except:
                break
        engTranslated.append(dialog)
    else:
        engTranslated.append([])

#save english translated buddy as a pickle object
pickle.dump(engTranslated, open('engTranslated.p', 'wb'))
