# for the lyrics scrape section
import requests
import time
from bs4 import BeautifulSoup
from collections import defaultdict, Counter
import os
import shutil

# Let's set up a dictionary of lists to hold our links
import random

artists = {'wallows':"https://www.azlyrics.com/w/wallows.html",
           'smash mouth':"https://www.azlyrics.com/s/smashmouth.html"} 

lyrics_pages = defaultdict(list)

for artist, artist_page in artists.items():
    # request the page and sleep
    r = requests.get(artist_page)
    time.sleep(5 + 10*random.random())

    # now extract the links to lyrics pages from this page
    # store the links `lyrics_pages` where the key is the artist and the
    # value is a list of links. 

    #pass along the HTML response to soup. in this case, we are looking for div class="listalbum-item" 
    soup = BeautifulSoup(r.text,'html.parser')    
    songs = soup.find_all("div", {"class": "listalbum-item"})

    #assign the resulting list to each artist.
    lyrics_pages[artist] = songs 


for artist, lp in lyrics_pages.items() :
    assert(len(set(lp)) > 20) 

# Let's see how long it's going to take to pull these lyrics 
# if we're waiting `5 + 10*random.random()` seconds 
for artist, links in lyrics_pages.items() : 
    print(f"For {artist} we have {len(links)}.")
    print(f"The full pull will take for this artist will take {round(len(links)*10/3600,2)} hours.")
    
def generate_filename_from_link(link) :
    
    if not link :
        return None
    
    # drop the http or https and the html
    name = link.replace("https","").replace("http","")
    name = link.replace(".html","")

    name = name.replace("/lyrics/","")
    
    # Replace useless characters with UNDERSCORE
    name = name.replace("://","").replace(".","_").replace("/","_")
    
    # tack on .txt
    name = name + ".txt"
    
    return(name)

# Make the lyrics folder here. If you'd like to practice your programming, add functionality 
# that checks to see if the folder exists. If it does, then use shutil.rmtree to remove it and create a new one.

if os.path.isdir("lyrics") : 
    shutil.rmtree("lyrics/")

os.mkdir("lyrics")

url_stub = "https://www.azlyrics.com" 
start = time.time()

total_pages = 0 

for artist in lyrics_pages:
    
    #check if we have a subfolder for this artist.
    artist_path = f'lyrics/{artist}/'
    if not os.path.isdir(artist_path):
        os.mkdir(artist_path)

    # 2. Iterate over the lyrics pages - our dictionary structure means we don't have to look for the song name later.
    for song in lyrics_pages[artist]:
        
        song_name = song.find('a').text 
        song_href = song.find('a').get('href')
        url = f'{url_stub}/{song_href}'
        
        # 3. Request the lyrics page. 
            # Don't forget to add a line like `time.sleep(5 + 10*random.random())`
            # to sleep after making the request
        
        r = requests.get(url)
        soup = BeautifulSoup(r.text,'html.parser')
        
        # 4. extract lyrics - title already exists.
        body = soup.find("div", {"class": "col-xs-12 col-lg-8 text-center"})
        
        # we observe that the fifth div inside of body contains the lyrics.
        lyrics = body.find_all("div")[5].text

        # create a file name.
        song_filename = f'{artist_path}{generate_filename_from_link(url)}'
        
        # 5. Write out the title, two returns ('\n'), and the lyrics. Use `generate_filename_from_url`
        #to generate the filename. 
        name_and_lyrics = f'{song_name}\n\n{lyrics}'

        with open(song_filename, 'w',encoding="utf-8") as f:
            f.write(str(name_and_lyrics))
        
        print(f'saved lyrics for {song_name} under {song_filename}.')
        #preview as a sanity check.
        print(f'preview: {lyrics[0:30]}')

        time.sleep(5+10*random.random())