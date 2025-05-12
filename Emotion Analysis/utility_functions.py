import json
from lyricsgenius import Genius
import configparser
import pandas as pd
import os
import librosa
import spacy
import re
import contractions
from nltk.tokenize import word_tokenize
from collections import Counter
import nltk
import numpy as np

nlp = spacy.load('en_core_web_sm')

def load_vad_lexicon(filepath):
    vad_df = pd.read_csv(filepath, sep='\t', header=None, names=['Word', 'Valence', 'Arousal', 'Dominance'])
    vad_lexicon = vad_df.set_index('Word')[['Valence', 'Arousal']].to_dict(orient='index')
    return vad_lexicon

def process_context(tokens):
    processed_tokens = []
    negate_next = False

    for i, token in enumerate(tokens):
        if token in ["not", "never", "don't", "can't"]:
            negate_next = True
        elif negate_next:
            processed_tokens.append(f"not_{token}")  # Append with "not_" prefix to indicate negation
            negate_next = False
        else:
            processed_tokens.append(token)
    return processed_tokens

def calculate_valence_arousal(tokens, vad_lexicon):
    total_valence = 0.0
    total_arousal = 0.0
    word_count = 0

    for token in tokens:
        if token.startswith("not_"):
            actual_token = token[4:]
            if actual_token in vad_lexicon:
                valence = 1.0 - vad_lexicon[actual_token]['Valence']
                arousal = vad_lexicon[actual_token]['Arousal']
                total_valence += valence
                total_arousal += arousal
                word_count += 1
        elif token in vad_lexicon:
            valence = vad_lexicon[token]['Valence']
            arousal = vad_lexicon[token]['Arousal']
            total_valence += valence
            total_arousal += arousal
            word_count += 1

    if word_count == 0:
        return 0.5, 0.5

    average_valence = total_valence / word_count
    average_arousal = total_arousal / word_count

    return average_valence, average_arousal

def calculate_row_valence_arousal(row, vad_lexicon):
    processed_tokens = process_context(row['Tokens'])
    average_valence, average_arousal = calculate_valence_arousal(processed_tokens, vad_lexicon)
    return pd.Series([average_valence, average_arousal], index=['Valence', 'Arousal'])

def extract_audio_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        mode = 'Major' if np.mean(chroma[0]) > np.mean(chroma[7]) else 'Minor'
        rms = librosa.feature.rms(y=y)
        loudness = np.mean(rms)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        danceability = np.std(onset_env)
        energy = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))

        return pd.Series([tempo, mode, loudness, danceability, energy])

    except Exception as e:
        print(f"Error processing file {audio_path}: {e}")
        return pd.Series([None, None, None, None, None])


def get_set_of_artists(data):
    artists = []
    for key, value in data.items():
        artists.append(value['artist'])
    return list(set(artists))


def filter_data_between_years(data, start_year, end_year):
    filtered_data = {}
    for key, value in data.items():
        year = int(value['date'])
        if start_year <= year < end_year:
            filtered_data[key] = value
    return filtered_data


def load_east_west_json():
    east_coast = load_east_coast_json()
    west_coast = load_west_coast_json()
    return east_coast, west_coast


def load_west_coast_json():
    return load_json('../../Data/rolling_stone_100_west_coast.json')


def load_east_coast_json():
    return load_json('../../Data/rolling_stone_100_east_coast.json')


def save_json(save_name, dictionary):
    with open(f'{save_name}', 'w') as json_file:
        json.dump(dictionary, json_file, ensure_ascii=False, indent=4)
    print(f'Data has been saved to "{save_name}"')


def load_json(file_name):
    with open(f'{file_name}', 'r') as json_file:
        loaded_data = json.load(json_file)
    return loaded_data


def lemmatize(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])


def cleanup_entity_rec(text):
    # Remove annotations in brackets and parentheses
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)

    # Normalize different quote marks to standard forms
    text = re.sub(r'[“”]', '"', text)
    text = re.sub(r'[‘’]', "'", text)

    # Clean up excessive use of punctuation marks
    text = re.sub(r'([.!?,:;])\1+', r'\1', text)
    text = re.sub(r'([.!?,:;])([^\s])', r'\1 \2', text)

    # Remove HTML entities and unnecessary special characters, standardize ampersands
    text = text.replace('&#8217;', '').replace('&amp;', '&')

    # Remove hashtags, unnecessary double quotes, and extra spaces
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\"(.*?)\"(.*?)\"(.*?)\"', r'"\1\2\3"', text)
    text = re.sub(r'\s+', ' ', text).strip()

    text = re.sub(r'\bCali\b', 'California', text)
    text = re.sub(r'\bNew York City\b', 'New York', text)
    text = re.sub(r'\bYork\b', 'New York', text)
    text = re.sub(r'\bNew New York\b', 'New York', text)
    text = re.sub(r'\bLA\b', 'Los Angeles', text)
    text = re.sub(r'\bVegas\b', 'Las Vegas', text)

    return text

def filter_tokens_by_document_frequency(df, column_name, min_doc_frequency=0.05, max_doc_frequency=0.85):
    """
    Filters tokens by their document frequency.

    :param df: DataFrame with a column of token lists.
    :param column_name: The name of the column containing the token lists.
    :param min_doc_frequency: Minimum percentage of documents a token should be in (0 to 1).
    :param max_doc_frequency: Maximum percentage of documents a token can be in (0 to 1).
    :return: Filtered list of tokens for each row in the DataFrame.
    """
    num_documents = len(df)

    # Flatten the token lists to count document frequency of each token
    token_document_count = Counter()
    for tokens in df[column_name]:
        unique_tokens_in_doc = set(tokens)  # Get unique tokens per document to count once per doc
        token_document_count.update(unique_tokens_in_doc)

    # Calculate document frequency thresholds
    min_docs = min_doc_frequency * num_documents
    max_docs = max_doc_frequency * num_documents

    # Create a list of tokens that meet the document frequency criteria
    filtered_tokens = {token for token, count in token_document_count.items() if min_docs <= count <= max_docs}

    # Filter the tokens in each document based on the criteria
    df[column_name] = df[column_name].apply(lambda tokens: [token for token in tokens if token in filtered_tokens])

    return df

def cleanup_for_sentiment_analysis(text):
    text = contractions.fix(text)

    # Basic cleanup
    text = re.sub(r"\[.*?\]", "", text)  # Remove text within brackets (e.g., [chorus])
    text = re.sub(r"\(.*?\)", "", text)  # Remove text within parentheses (e.g., (verse))
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'[“”]', '"', text)  # Normalize double quotation marks
    text = re.sub(r'[‘’]', "'", text)  # Normalize apostrophes

    # Replace slang terms with their standard equivalents using a dictionary
    slang_dict = {
        r'\bmoth(?:a|e|er)?(?:f(?:uck|uk|unk|ck|uc|uck|ucker))?(?:a|as|az|ez|ers?|erz?|in|ing|ering|err?s?|ershit|ershit|ingshit|uckin|uckering|uckerrs|s)?\b': 'motherfucker',
        r'\bniggas?\b': 'nigga',
        r'\bniggas\b': 'nigga',
        r'\bniggaa\b': 'nigga',
        r'\bactin\b': 'act',
        r'\bniggaz\b': 'nigga',
        r'\bblunts\b': 'blunt',
        r'\bbodies\b': 'body',
        r'\bboots\b': 'boot',
        r'\bboys\b': 'boy',
        r'\bcalled\b': 'call',
        r'\bcars\b': 'car',
        r'\bcats\b': 'cat',
        r'\bchanged\b': 'change',
        r'\bcmon\b': 'come on',
        r'\bcomes\b': 'come',
        r'\bcuts\b': 'cut',
        r'\bdropped\b': 'drop',
        r'\bears\b': 'ear',
        r'\bends\b': 'end',
        r'\benemies\b': 'enemy',
        r'\bniggaboo\b': 'nigga',
        r'\bnigg(?:ar|ro)es\b': 'nigga',
        r'\bnigg(?:ers|uz|ys|gie|gy)\b': 'nigga',
        r'\bniga\b': 'nigga',
        r'\bnigas\b': 'nigga',
        r'\bnigg\b': 'nigga',
        r'\byo\b': 'yeah',
        r'\baiyyo\b': 'yeah',
        r'\bayo\b': 'yeah',
        r'\baight\b': 'alright',
        r'\bbitches\b': 'bitch',
        r'\bbrains\b': 'bitch',
        r'\bbreakin\b': 'break',
        r'\bbullets\b': 'bullet',
        r'\bcheckin\b': 'check',
        r'\byah\b': 'yeah',
        r'\bya\b': 'yeah',
        r'\byea\b': 'yeah',
        r'\bblowin\b': 'blow',
        r'\bworkin\b': 'work',
        r'\bwatchin\b': 'watch',
        r'\bwalkin\b': 'walk',
        r'\bwaitin\b': 'wait',
        r'\bbustin\b': 'bust',
        r'\btryin\b': 'try',
        r'\btrippin\b': 'trip',
        r'\bcallin\b': 'call',
        r'\bchillin\b': 'chill',
        r'\bcomin\b': 'come',
        r'\bcoming\b': 'come',
        r'\byep\b': 'yeah',
        r'\bcops\b': 'cop',
        r'\bdogg\b': 'dog',
        r'\bcrews\b': 'crew',
        r'\bdied\b': 'die',
        r'\bdogs\b': 'dog',
        r'\bdollars\b': 'dollar',
        r'\bdreams\b': 'dream',
        r'\bdrugs\b': 'drug',
        r'\bfacts\b': 'fact',
        r'\beyes\b': 'eye',
        r'\bdays\b': 'day',
        r'\bnuts\b': 'nut',
        r'\bok\b': 'okay',
        r'\bones\b': 'one',
        r'\bpeoples\b': 'people',
        r'\bplayed\b': 'play',
        r'\bplaying\b': 'play',
        r'\bplayers\b': 'play',
        r'\bpockets\b': 'pocket',
        r'\bpoppin\b': 'pop',
        r'\bpops\b': 'pop',
        r'\bpulled\b': 'pull',
        r'\bpumpin\b': 'pump',
        r'\bpunks\b': 'punk',
        r'\bputtin\b': 'put',
        r'\braised\b': 'raise',
        r'\brappers\b': 'rap',
        r'\brappin\b': 'rap',
        r'\brecords\b': 'record',
        r'\brhymes\b': 'rhyme',
        r'\brockin\b': 'rock',
        r'\brules\b': 'rule',
        r'\brunnin\b': 'run',
        r'\brunning\b': 'run',
        r'\bsaying\b': 'say',
        r'\bsays\b': 'say',
        r'\bseems\b': 'seem',
        r'\bshits\b': 'shit',
        r'\bshorty\b': 'short',
        r'\bsmoked\b': 'smoke',
        r'\bsmoking\b': 'smoke',
        r'\bsongs\b': 'song',
        r'\bsounds\b': 'sound',
        r'\bstarted\b': 'start',
        r'\bstarts\b': 'start',
        r'\bstreets\b': 'street',
        r'\bsuckers\b': 'sucker',
        r'\btakes\b': 'take',
        r'\btakin\b': 'take',
        r'\btaking\b': 'take',
        r'\btha\b': 'the',
        r'\bthoughts\b': 'thought',
        r'\bthugs\b': 'thug',
        r'\btimes\b': 'time',
        r'\btracks\b': 'track',
        r'\btricks\b': 'trick',
        r'\bused\b': 'use',
        r'\bways\b': 'way',
        r'\bwomen\b': 'woman',
        r'\bwords\b': 'word',
        r'\byears\b': 'year',
        r'\bfeelin\b': 'feel',
        r'\bfeeling\b': 'feel',
        r'\bfingers\b': 'finger',
        r'\bfools\b': 'fool',
        r'\bgoes\b': 'go',
        r'\bgoin\b': 'go',
        r'\bgoing\b': 'go',
        r'\bhangin\b': 'hang',
        r'\bhittin\b': 'hit',
        r'\bholdin\b': 'hold',
        r'\bhomies\b': 'homie',
        r'\bhomeboy\b': 'homie',
        r'\bhomeboys\b': 'homie',
        r'\bflippin\b': 'flip',
        r'\bdoin\b': 'doing',
        r'\bdrinkin\b': 'drink',
        r'\bfreaks\b': 'freak',
        r'\bfucked\b': 'fuck',
        r'\bfuckin\b': 'fuck',
        r'\bgettin\b': 'get',
        r'\bgetting\b': 'get',
        r'\bfucking\b': 'fuck',
        r'\bkeepin\b': 'keep',
        r'\bkicked\b': 'kick',
        r'\bkickin\b': 'kick',
        r'\bkeys\b': 'key',
        r'\bkilled\b': 'kill',
        r'\bkillin\b': 'kill',
        r'\bkills\b': 'kill',
        r'\bknowin\b': 'know',
        r'\bnothin\b': 'nothing',
        r'\bnuttin\b': 'nothing',
        r'\bplayin\b': 'playing',
        r'\brings\b': 'ring',
        r'\bridin\b': 'ride',
        r'\brollin\b': 'roll',
        r'\brolling\b': 'roll',
        r'\brolled\b': 'roll',
        r'\brippin\b': 'rip',
        r'\bsayin\b': 'saying',
        r'\bscreamin\b': 'scream',
        r'\bseein\b': 'see',
        r'\bsellin\b': 'sell',
        r'\bshootin\b': 'shoot',
        r'\bshots\b': 'shot',
        r'\bsippin\b': 'sip',
        r'\bsittin\b': 'sit',
        r'\bsmokin\b': 'smoke',
        r'\bsomethin\b': 'something',
        r'\btalking\b': 'talk',
        r'\bthings\b': 'thing',
        r'\bthinking\b': 'think',
        r'\btrying\b': 'try',
        r'\bwalked\b': 'walk',
        r'\bwalking\b': 'walk',
        r'\bknown\b': 'know',
        r'\bknows\b': 'know',
        r'\bladies\b': 'lady',
        r'\blayin\b': 'lay',
        r'\bleavin\b': 'leave',
        r'\blines\b': 'line',
        r'\blives\b': 'live',
        r'\blivin\b': 'live',
        r'\bliving\b': 'live',
        r'\blooked\b': 'look',
        r'\blookin\b': 'look',
        r'\blooking\b': 'look',
        r'\blooks\b': 'look',
        r'\bmakes\b': 'make',
        r'\bmakin\b': 'make',
        r'\bmaking\b': 'make',
        r'\bmoves\b': 'move',
        r'\bmovin\b': 'move',
        r'\bgivin\b': 'give',
        r'\bgiving\b': 'give',
        r'\bgangsta\b': 'gangster',
        r'\bthrowin\b': 'throw',
        r'\bthinkin\b': 'think',
        r'\btellin\b': 'tell',
        r'\btalkin\b': 'talk',
        r'\bsteppin\b': 'step',
        r'\bstandin\b': 'stand',
        r'\bdroppin\b': 'drop',
        r'\byeahhu\b': 'yeah',
        r'\bbrothers\b': 'brother',
        r'\bbein\b': 'be',
        r'\byeahhur\b': 'yeah',
        r'\byeahh+\b': 'yeah',
        r'\bdoggz\b': 'dogs',
        r'\bgonna\b': 'going to',
        r'\bwanna\b': 'want to',
        r'\bcuz\b': 'because',
        r'\bcoz\b': 'because',
        r'\bcause\b': 'because',
        r'\bkinda\b': 'kind of',
        r'\bgotta\b': 'got to',
        r'\boutta\b': 'out of',
        r'\blotta\b': 'lot of',
        r'\blemme\b': 'let me',
        r'\bcmon\b': 'come on',
        r'\bfiends\b': 'fiend',
        r'\bflows\b': 'flow',
        r'\bgames\b': 'game',
        r'\bgirls\b': 'girl',
        r'\bgots\b': 'got',
        r'\bguns\b': 'gun',
        r'\bguys\b': 'guy',
        r'\bhands\b': 'hand',
        r'\bhappened\b': 'happen',
        r'\bheaded\b': 'head',
        r'\bheads\b': 'head',
        r'\bhits\b': 'hit',
        r'\bhoes\b': 'hoe',
        r'\bkeeps\b': 'keep',
        r'\bknocked\b': 'knock',
        r'\blights\b': 'light',
        r'\bmans\b': 'man',
        r'\bmcs\b': 'mc',
        r'\bmeans\b': 'mean',
        r'\bmics\b': 'mic',
        r'\bminds\b': 'mind',
        r'\bmomma\b': 'mom',
        r'\bmoms\b': 'mom',
        r'\bgimme\b': 'give me',
        r'\bain\'t\b': 'is not',
        r'\bimma\b': 'i am going to',
        r'\baccordin\b': 'according',
        r'\bacapella\b': 'a cappella',
        r'\baand\b': 'and',
        r'\baant\b': 'want',
    }

    # Replace slang terms using word boundaries
    for slang, standard in slang_dict.items():
        text = re.sub(slang, standard, text, flags=re.IGNORECASE)

    # Remove repeated punctuations, keeping single punctuation intact
    text = re.sub(r'([.!?,:;])\1+', r'\1', text)

    # Remove special characters except for apostrophes, since "don't" is valid
    text = re.sub(r'[^a-zA-Z0-9\'\s]', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Convert to lowercase to match dictionary entries
    text = text.lower()

    # Tokenization
    tokens = word_tokenize(text)

    # Avoid stemming or aggressive lemmatization here to ensure words match lexicon entries.
    # For instance, don't convert "running" to "run" unless the lexicon expects the base form.

    # Remove tokens with numbers (keep only alphabetic tokens)
    tokens = [token for token in tokens if token.isalpha()]

    # Remove single letter tokens unless they are pronouns (e.g., "I")
    tokens = [token for token in tokens if len(token) > 1 or token.lower() == 'i']

    # Stopwords removal (optional, be careful as stopwords like "not" are crucial)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    custom_stop_words = {'yeah', 'uh', 'oh', 'like', 'as', 'bo', 'ab', 'aa', "aaghh", "aah", "aahh", "aaooww", "aaw", "acabe",
                         "abster", "ah", 'boo', 'da', 'de', 'fo', 'huh', 'ha', 'ho', 'mo', 'ooh', 'ta', 'uhh'}
    stop_words = stop_words.union(custom_stop_words)
    tokens = [token for token in tokens if token.lower() not in stop_words]

    # Reconstruct cleaned text from tokens
    cleaned_text = ' '.join(tokens)

    return cleaned_text


def cleanup(text):
    # Expand contractions
    text = contractions.fix(text)

    # Original cleanup steps
    text = re.sub(r"\[.*?\]", "", text)  # Remove text within brackets
    text = re.sub(r"\(.*?\)", "", text)  # Remove text within parentheses
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'[“”]', '"', text)  # Normalize quotation marks
    text = re.sub(r'[‘’]', "'", text)  # Normalize apostrophes
    text = re.sub(r'([.!?,:;])\1+', r'\1', text)  # Remove repeated punctuation
    text = re.sub(r'([.!?,:;])([^\s])', r'\1 \2', text)  # Add space after punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    text = text.lower()  # Convert to lowercase
    text = text.strip('"')
    text = text.replace('.', '').replace('-', ' ').replace("’", '')
    text = text.replace("?", '').replace("!", '').replace("*", 'i')
    text = text.replace('&#8217;', '').replace(',', '').replace('&amp;', '&')
    text = text.replace('\n', '')

    slang_dict = {
        r'\bmoth(?:a|e|er)?(?:f(?:uck|uk|unk|ck|uc|uck|ucker))?(?:a|as|az|ez|ers?|erz?|in|ing|ering|err?s?|ershit|ershit|ingshit|uckin|uckering|uckerrs|s)?\b': 'motherfucker',
        r'\bniggas?\b': 'nigga',
        r'\bniggas\b': 'nigga',
        r'\bniggaa\b': 'nigga',
        r'\bactin\b': 'act',
        r'\bniggaz\b': 'nigga',
        r'\bblunts\b': 'blunt',
        r'\bbodies\b': 'body',
        r'\bboots\b': 'boot',
        r'\bboys\b': 'boy',
        r'\bcalled\b': 'call',
        r'\bcars\b': 'car',
        r'\bcats\b': 'cat',
        r'\bchanged\b': 'change',
        r'\bcmon\b': 'come on',
        r'\bcomes\b': 'come',
        r'\bcuts\b': 'cut',
        r'\bdropped\b': 'drop',
        r'\bears\b': 'ear',
        r'\bends\b': 'end',
        r'\benemies\b': 'enemy',
        r'\bniggaboo\b': 'nigga',
        r'\bnigg(?:ar|ro)es\b': 'nigga',
        r'\bnigg(?:ers|uz|ys|gie|gy)\b': 'nigga',
        r'\bniga\b': 'nigga',
        r'\bnigas\b': 'nigga',
        r'\bnigg\b': 'nigga',
        r'\byo\b': 'yeah',
        r'\baiyyo\b': 'yeah',
        r'\bayo\b': 'yeah',
        r'\baight\b': 'alright',
        r'\bbitches\b': 'bitch',
        r'\bbrains\b': 'bitch',
        r'\bbreakin\b': 'break',
        r'\bbullets\b': 'bullet',
        r'\bcheckin\b': 'check',
        r'\byah\b': 'yeah',
        r'\bya\b': 'yeah',
        r'\byea\b': 'yeah',
        r'\bblowin\b': 'blow',
        r'\bworkin\b': 'work',
        r'\bwatchin\b': 'watch',
        r'\bwalkin\b': 'walk',
        r'\bwaitin\b': 'wait',
        r'\bbustin\b': 'bust',
        r'\btryin\b': 'try',
        r'\btrippin\b': 'trip',
        r'\bcallin\b': 'call',
        r'\bchillin\b': 'chill',
        r'\bcomin\b': 'come',
        r'\bcoming\b': 'come',
        r'\byep\b': 'yeah',
        r'\bcops\b': 'cop',
        r'\bdogg\b': 'dog',
        r'\bcrews\b': 'crew',
        r'\bdied\b': 'die',
        r'\bdogs\b': 'dog',
        r'\bdollars\b': 'dollar',
        r'\bdreams\b': 'dream',
        r'\bdrugs\b': 'drug',
        r'\bfacts\b': 'fact',
        r'\beyes\b': 'eye',
        r'\bdays\b': 'day',
        r'\bnuts\b': 'nut',
        r'\bok\b': 'okay',
        r'\bones\b': 'one',
        r'\bpeoples\b': 'people',
        r'\bplayed\b': 'play',
        r'\bplaying\b': 'play',
        r'\bplayers\b': 'play',
        r'\bpockets\b': 'pocket',
        r'\bpoppin\b': 'pop',
        r'\bpops\b': 'pop',
        r'\bpulled\b': 'pull',
        r'\bpumpin\b': 'pump',
        r'\bpunks\b': 'punk',
        r'\bputtin\b': 'put',
        r'\braised\b': 'raise',
        r'\brappers\b': 'rap',
        r'\brappin\b': 'rap',
        r'\brecords\b': 'record',
        r'\brhymes\b': 'rhyme',
        r'\brockin\b': 'rock',
        r'\brules\b': 'rule',
        r'\brunnin\b': 'run',
        r'\brunning\b': 'run',
        r'\bsaying\b': 'say',
        r'\bsays\b': 'say',
        r'\bseems\b': 'seem',
        r'\bshits\b': 'shit',
        r'\bshorty\b': 'short',
        r'\bsmoked\b': 'smoke',
        r'\bsmoking\b': 'smoke',
        r'\bsongs\b': 'song',
        r'\bsounds\b': 'sound',
        r'\bstarted\b': 'start',
        r'\bstarts\b': 'start',
        r'\bstreets\b': 'street',
        r'\bsuckers\b': 'sucker',
        r'\btakes\b': 'take',
        r'\btakin\b': 'take',
        r'\btaking\b': 'take',
        r'\btha\b': 'the',
        r'\bthoughts\b': 'thought',
        r'\bthugs\b': 'thug',
        r'\btimes\b': 'time',
        r'\btracks\b': 'track',
        r'\btricks\b': 'trick',
        r'\bused\b': 'use',
        r'\bways\b': 'way',
        r'\bwomen\b': 'woman',
        r'\bwords\b': 'word',
        r'\byears\b': 'year',
        r'\bfeelin\b': 'feel',
        r'\bfeeling\b': 'feel',
        r'\bfingers\b': 'finger',
        r'\bfools\b': 'fool',
        r'\bgoes\b': 'go',
        r'\bgoin\b': 'go',
        r'\bgoing\b': 'go',
        r'\bhangin\b': 'hang',
        r'\bhittin\b': 'hit',
        r'\bholdin\b': 'hold',
        r'\bhomies\b': 'homie',
        r'\bhomeboy\b': 'homie',
        r'\bhomeboys\b': 'homie',
        r'\bflippin\b': 'flip',
        r'\bdoin\b': 'doing',
        r'\bdrinkin\b': 'drink',
        r'\bfreaks\b': 'freak',
        r'\bfucked\b': 'fuck',
        r'\bfuckin\b': 'fuck',
        r'\bgettin\b': 'get',
        r'\bgetting\b': 'get',
        r'\bfucking\b': 'fuck',
        r'\bkeepin\b': 'keep',
        r'\bkicked\b': 'kick',
        r'\bkickin\b': 'kick',
        r'\bkeys\b': 'key',
        r'\bkilled\b': 'kill',
        r'\bkillin\b': 'kill',
        r'\bkills\b': 'kill',
        r'\bknowin\b': 'know',
        r'\bnothin\b': 'nothing',
        r'\bnuttin\b': 'nothing',
        r'\bplayin\b': 'playing',
        r'\brings\b': 'ring',
        r'\bridin\b': 'ride',
        r'\brollin\b': 'roll',
        r'\brolling\b': 'roll',
        r'\brolled\b': 'roll',
        r'\brippin\b': 'rip',
        r'\bsayin\b': 'saying',
        r'\bscreamin\b': 'scream',
        r'\bseein\b': 'see',
        r'\bsellin\b': 'sell',
        r'\bshootin\b': 'shoot',
        r'\bshots\b': 'shot',
        r'\bsippin\b': 'sip',
        r'\bsittin\b': 'sit',
        r'\bsmokin\b': 'smoke',
        r'\bsomethin\b': 'something',
        r'\btalking\b': 'talk',
        r'\bthings\b': 'thing',
        r'\bthinking\b': 'think',
        r'\btrying\b': 'try',
        r'\bwalked\b': 'walk',
        r'\bwalking\b': 'walk',
        r'\bknown\b': 'know',
        r'\bknows\b': 'know',
        r'\bladies\b': 'lady',
        r'\blayin\b': 'lay',
        r'\bleavin\b': 'leave',
        r'\blines\b': 'line',
        r'\blives\b': 'live',
        r'\blivin\b': 'live',
        r'\bliving\b': 'live',
        r'\blooked\b': 'look',
        r'\blookin\b': 'look',
        r'\blooking\b': 'look',
        r'\blooks\b': 'look',
        r'\bmakes\b': 'make',
        r'\bmakin\b': 'make',
        r'\bmaking\b': 'make',
        r'\bmoves\b': 'move',
        r'\bmovin\b': 'move',
        r'\bgivin\b': 'give',
        r'\bgiving\b': 'give',
        r'\bgangsta\b': 'gangster',
        r'\bthrowin\b': 'throw',
        r'\bthinkin\b': 'think',
        r'\btellin\b': 'tell',
        r'\btalkin\b': 'talk',
        r'\bsteppin\b': 'step',
        r'\bstandin\b': 'stand',
        r'\bdroppin\b': 'drop',
        r'\byeahhu\b': 'yeah',
        r'\bbrothers\b': 'brother',
        r'\bbein\b': 'be',
        r'\byeahhur\b': 'yeah',
        r'\byeahh+\b': 'yeah',
        r'\bdoggz\b': 'dogs',
        r'\bgonna\b': 'going to',
        r'\bwanna\b': 'want to',
        r'\bcuz\b': 'because',
        r'\bcoz\b': 'because',
        r'\bcause\b': 'because',
        r'\bkinda\b': 'kind of',
        r'\bgotta\b': 'got to',
        r'\boutta\b': 'out of',
        r'\blotta\b': 'lot of',
        r'\blemme\b': 'let me',
        r'\bcmon\b': 'come on',
        r'\bfiends\b': 'fiend',
        r'\bflows\b': 'flow',
        r'\bgames\b': 'game',
        r'\bgirls\b': 'girl',
        r'\bgots\b': 'got',
        r'\bguns\b': 'gun',
        r'\bguys\b': 'guy',
        r'\bhands\b': 'hand',
        r'\bhappened\b': 'happen',
        r'\bheaded\b': 'head',
        r'\bheads\b': 'head',
        r'\bhits\b': 'hit',
        r'\bhoes\b': 'hoe',
        r'\bkeeps\b': 'keep',
        r'\bknocked\b': 'knock',
        r'\blights\b': 'light',
        r'\bmans\b': 'man',
        r'\bmcs\b': 'mc',
        r'\bmeans\b': 'mean',
        r'\bmics\b': 'mic',
        r'\bminds\b': 'mind',
        r'\bmomma\b': 'mom',
        r'\bmoms\b': 'mom',
        r'\bgimme\b': 'give me',
        r'\bain\'t\b': 'is not',
        r'\bimma\b': 'i am going to',
        r'\baccordin\b': 'according',
        r'\bacapella\b': 'a cappella',
        r'\baand\b': 'and',
        r'\baant\b': 'want',
    }

    # Replace slang terms using word boundaries
    for slang, standard in slang_dict.items():
        text = re.sub(slang, standard, text, flags=re.IGNORECASE)

    # Remove remaining special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove tokens with numbers
    tokens = [token for token in tokens if not any(char.isdigit() for char in token)]

    # Remove single letter tokens
    tokens = [token for token in tokens if len(token) > 1]

    # lemmatizer = WordNetLemmatizer()
    # tokens = [lemmatizer.lemmatize(token) for token in tokens]

    tokens = [re.sub(r'(.)\1{2,}', r'\1\1', token) for token in tokens]

    stop_words = set(nltk.corpus.stopwords.words('english'))
    custom_stop_words = {'yeah', 'uh', 'oh', 'like', 'as', 'bo', 'ab', 'aa', "aaghh", "aah", "aahh", "aaooww", "aaw", "acabe",
                         "abster", "ah", 'boo', 'da', 'de', 'fo', 'huh', 'ha', 'ho', 'mo', 'ooh', 'ta', 'uhh'}
    stop_words = stop_words.union(custom_stop_words)
    tokens = [token for token in tokens if token.lower() not in stop_words]

    # Reconstruct the text
    cleaned_text = ' '.join(tokens)

    return tokens, cleaned_text


def light_preprocessing(text):
    text = re.sub(r"\[.*?\]", "", text)  # Remove text within brackets
    text = re.sub(r"\(.*?\)", "", text)  # Remove text within parentheses
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'[“”]', '"', text)  # Normalize quotation marks
    text = re.sub(r'[‘’]', "'", text)  # Normalize apostrophes
    text = re.sub(r'([.!?,:;])\1+', r'\1', text)  # Remove repeated punctuation
    text = re.sub(r'([.!?,:;])([^\s])', r'\1 \2', text)  # Add space after punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    text = text.lower()
    text = text.strip('"')
    text = text.replace('.', '').replace('-', ' ').replace("’", '')
    text = text.replace("?", '').replace("!", '').replace("*", 'i')
    text = text.replace('&#8217;', '').replace(',', '').replace('&amp;', '&')
    text = text.replace('\n', '')
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def artist_cleanup(text):
    _text = text
    if 'feat.' in _text:
        _text = _text.split('feat.')[0]
    _text = cleanup(_text)
    return _text


def get_genius_object():
    config = configparser.ConfigParser()
    # Load the config file
    config.read('./config.ini')  # Update with the correct path to your config file
    # Get the API key
    client_access_token = config['API']['api_key']
    return Genius(client_access_token)


# --------------------------------DATAFRAME RELATED FUNCTIONS----------------------------------------
def load_txt_into_dataframe(path):
    """
    Input: path - string
    Function to recursively go trought on the folder structure
    and load all the .txt files into a dataframe.
    Used to load lyrics.
    """

    base_directory = path
    year_pattern = r'\(\d{4}\)'
    # Initialize a list to store data
    data = []

    # Loop through each root, directory, and file in the base directory
    for root, dirs, files in os.walk(base_directory):
        for file in files:
            # Check if the file is a text file
            if file.endswith(".txt"):
                # Construct the full file path
                file_path = os.path.join(root, file)
                # Open and read the contents of the text file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    match = re.search(year_pattern, file_path)
                    if match:
                        extracted_year = match.group()
                # Append the file path, file name, and content to the data list
                data.append({"Coast": file_path.split('/')[-4], "Artist": file_path.split('/')[-3],
                             "Album": file_path.split('/')[-2],
                             "Album Release Year": int(extracted_year[1:-1]),
                             "Song": file.replace('.txt', ''), "Lyrics": content})

    return pd.DataFrame(data)


def load_audio_into_dataframe(path):
    """
    Input: path - string
    Function to recursively go through the folder structure
    and load metadata from all the audio files into a DataFrame.
    Used to load audio file metadata.
    """

    base_directory = path
    data = []

    # Loop through each root, directory, and file in the base directory
    for root, dirs, files in os.walk(base_directory):
        for file in files:
            # Check if the file is an audio file (e.g., .wav, .mp3)
            if file.lower().endswith((".wav", ".mp3")):
                # Construct the full file path
                file_path = os.path.join(root, file)
                # Load the audio file with librosa
                y, sr = librosa.load(file_path, sr=None)  # sr=None to preserve the native sampling rate

                # Get duration in seconds
                duration = librosa.get_duration(y=y, sr=sr)

                # Append the file path, file name, duration, and sample rate to the data list
                data.append({
                    "FilePath": file_path,
                    "FileName": file,
                    "Duration (s)": duration,
                    "Sample Rate (Hz)": sr,
                    # You can add more features here
                })

    # Convert the list of dictionaries into a pandas DataFrame
    return pd.DataFrame(data)


def calculate_concreteness_score(word_scores):
    nominator = 0
    denomiantor = 0
    for key, value in word_scores.items():
        nominator += value[0] * value[1]
        denomiantor += value[1]

    return nominator / denomiantor


def calculate_correctness_score_of_tokens(dataframe, concreteness_ratings):
    for index, row in dataframe.iterrows():
        # Calculate new value for the current row
        frequency_distribution = {word: row['Tokens'].count(word) for word in set(row['Tokens'])}
        word_scores = {word: (concreteness_ratings.get(word, 0), freq) for word, freq in frequency_distribution.items()}
        correctness_score = calculate_concreteness_score(word_scores)
        # Assign the new value to a new column for that row
        dataframe.at[index, 'Correctness'] = correctness_score
    return dataframe


def word_count_of_text(text):
    return len(text.split(' '))


def unique_word_count_of_text(text):
    return len(list(set(text.split(' '))))


def filter_dataframe_by_artist(df, artist):
    return df[df['Artist'] == artist]


def filter_dataframe_by_album(df, year):
    return df[df['Album Release Year'] == year]


# --------------------------------DATAFRAME RELATED FUNCTIONS----------------------------------------

# --------------------------------LOADER FUNCTIONS----------------------------------------
def get_all_artists(json_path):
    with open(json_path) as json_file:
        data = json.load(json_file)
    return data


def get_all_lyrics_of_an_artist(artist_name, json_path):
    artists_data = get_all_artists(json_path)  # Assuming this function returns a dict-like object
    artist_data = artists_data[artist_name]

    lyrics_paths = [item['lyrics_path'] for item in artist_data if 'lyrics_path' in item]

    # Collect all DataFrames in a list first
    dfs = [load_txt_into_dataframe(path) for path in lyrics_paths]

    # Concatenate all DataFrames at once
    all_lyrics_df = pd.concat(dfs, ignore_index=True)

    return all_lyrics_df


def get_all_lyrics_of_an_artist_between_years(artist_name, json_path, start_year, end_year):
    artists_data = get_all_artists(json_path)  # Assuming this function returns a dict-like object
    artist_data = artists_data[artist_name]

    lyrics_paths = []

    for item in artist_data:
        if 'lyrics_path' in item:
            release_year = int(item['release_date'])
            if start_year <= release_year <= end_year:
                lyrics_paths.append(item['lyrics_path'])

    # Collect all DataFrames in a list first
    dfs = [load_txt_into_dataframe(path) for path in lyrics_paths]

    # Concatenate all DataFrames at once
    all_lyrics_df = pd.concat(dfs, ignore_index=True)

    return all_lyrics_df


def get_all_audio_of_an_artist(artist_name, json_path):
    artists_data = get_all_artists(json_path)  # Assuming this function returns a dict-like object
    artist_data = artists_data[artist_name]

    audio_paths = [item['audio_path'] for item in artist_data if 'audio_path' in item]

    # Collect all DataFrames in a list first
    dfs = [load_audio_into_dataframe(path) for path in audio_paths]

    # Concatenate all DataFrames at once
    all_audio_df = pd.concat(dfs, ignore_index=True)

    return all_audio_df


def get_all_audio_of_an_artist_between_years(artist_name, json_path, start_year, end_year):
    artists_data = get_all_artists(json_path)  # Assuming this function returns a dict-like object
    artist_data = artists_data[artist_name]

    audio_paths = []

    for item in artist_data:
        if 'audio_path' in item:
            release_year = int(item['release_date'])
            if start_year <= release_year <= end_year:
                audio_paths.append(item['audio_path'])

    # Collect all DataFrames in a list first
    dfs = [load_audio_into_dataframe(path) for path in audio_paths]

    # Concatenate all DataFrames at once
    all_audio_df = pd.concat(dfs, ignore_index=True)

    return all_audio_df


def get_all_artist_lyrics_between_years(json_path, start_year, end_year):
    artists_data = get_all_artists(json_path)

    lyrics_paths = []

    for item in artists_data:
        if 'lyrics_path' in item:
            release_year = int(item['release_date'])
            if start_year <= release_year <= end_year:
                lyrics_paths.append(item['lyrics_path'])

    dfs = [load_txt_into_dataframe(path) for path in lyrics_paths]
    all_lyrics_df = pd.concat(dfs, ignore_index=True)

    return all_lyrics_df


def get_all_artist_lyrics(json_path):
    artists_data = get_all_artists(json_path)

    lyrics_paths = []

    for item in artists_data.values():
        for i in item:
            if 'lyrics_path' in i:
                lyrics_paths.append(i['lyrics_path'])

    dfs = [load_txt_into_dataframe(path) for path in lyrics_paths]
    all_lyrics_df = pd.concat(dfs, ignore_index=True)

    return all_lyrics_df


def get_all_artist_audio_between_years(json_path, start_year, end_year):
    artists_data = get_all_artists(json_path)

    audio_paths = []

    for item in artists_data:
        if 'audio_path' in item:
            release_year = int(item['release_date'])
            if start_year <= release_year <= end_year:
                audio_paths.append(item['audio_paths'])

    dfs = [load_audio_into_dataframe(path) for path in audio_paths]
    all_audio_df = pd.concat(dfs, ignore_index=True)

    return all_audio_df


def get_all_artist_audio(json_path):
    artists_data = get_all_artists(json_path)

    audio_paths = []

    for item in artists_data.values():
        for i in item:
            if 'audio_path' in i:
                audio_paths.append(i['audio_path'])

    dfs = [load_audio_into_dataframe(path) for path in audio_paths]
    all_audio_df = pd.concat(dfs, ignore_index=True)

    return all_audio_df
