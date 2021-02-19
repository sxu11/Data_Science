from pprint import pprint
from googleapiclient.discovery import build
import os

YOUTUBE_API_KEY = os.environ.get('YOUTUBE_API_KEY')
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

lis = ['UCeLHszkByNZtPKcaVXOCOQQ']
for i in lis:
    channels_response = youtube.channels().list(part='statistics', id=i).execute()
    print(i, channels_response['items'][0]['statistics']['videoCount'])
for i in lis:
    channels_response = youtube.channels().list(part='contentDetails', id=i).execute()
    for channel in channels_response['items']:
        uploads_list_id = channel["contentDetails"]["relatedPlaylists"]["uploads"]
        playlistitems_list_request = youtube.playlistItems().list(
            playlistId=uploads_list_id,
            part="snippet",
            maxResults=50
          )
        while playlistitems_list_request:
            playlistitems_list_response = playlistitems_list_request.execute()
            for playlist_item in playlistitems_list_response["items"]:
                # pprint(playlist_item)
                title = playlist_item["snippet"]["title"]
                video_id = playlist_item["snippet"]["resourceId"]["videoId"]
                print(title, video_id)
            playlistitems_list_request = youtube.playlistItems().list_next(
                playlistitems_list_request, playlistitems_list_response
            )