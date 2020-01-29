import json
import requests

def post(dic,url):

    headers = {
        "Content-Type" : 'application/json'
    }
    requests.post(url,headers=headers,data=json.dumps(dic))

class discord_webhook:
    
    def post(self,content):
        post_dec = {
            "content": content,
            "username": self.username
        }

        post(post_dec,self.url)

    def post_dec(self,dec):
        post_dec = {
            "username":self.username
        }
        post_dec.update(dec)
        post(post_dec,self.url)
    

    def __init__(self,settings):
        
        with open(settings) as f:
            s = f.read()
            json_data = json.loads(s)

        self.url = json_data["url"]
        self.username = json_data["username"]