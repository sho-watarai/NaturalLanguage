import json
import os
import re
import reprlib
import tweepy
import time

from datetime import datetime

consumer_key = "your_consumer_key"
consumer_secret = "your_consumer_secret"
access_token = "your_access_token"
access_token_secret = "your_access_token_secret"


class StreamListenerSTSA(tweepy.StreamListener):
    def __init__(self, api):
        super(StreamListenerSTSA, self).__init__()
        self.api = api
        self.batch_size = 100
        self.num_handled = 0
        self.queue = []

    def on_data(self, data):
        """ Routes the raw stream data to the appropriate method """
        raw = json.loads(data)
        if "in_reply_to_status_id" in raw:
            if self.on_status(raw) is False:
                return False
        elif "limit" in raw:
            if self.on_limit(raw["limit"]["track"]) is False:
                return False
        return True

    def on_status(self, raw):
        if isinstance(raw.get("in_reply_to_status_id"), int):
            print("{:>3d} : {}".format(len(self.queue), reprlib.repr(raw["text"])))
            line = (raw.get("in_reply_to_status_id"), raw.get("text"))
            self.queue.append(line)
            if len(self.queue) >= self.batch_size:
                self.dump()

        return True

    def on_error(self, status):
        print("ON ERROR:", status)

    def on_limit(self, track):
        print("ON LIMIT:", track)

    def dump(self):
        pcnt = 0
        with open("./twitter/%s.txt" % (datetime.now().strftime("%Y%m%d_%H%M%S")), "w", encoding="UTF-8") as f:
            (sids, texts), self.queue = zip(*self.queue), []

            while True:
                try:
                    lines_mapper = {s.id_str: s.text for s in self.api.statuses_lookup(sids)}
                    break
                except Exception as e:
                    print(e)
                    time.sleep(10)

            lines_grps = [[lines_mapper.get(str(sid)), txt] for sid, txt in zip(sids, texts) if lines_mapper.get(str(sid))]
            lines_grps = [[self.preprocess(s) for s in lines] for lines in lines_grps]

            for lines in lines_grps:
                for i in range(len(lines) - 1):
                    f.write("%s\n%s\n" % (lines[i], lines[i + 1]))
                    pcnt += 1

        self.num_handled += pcnt

    def preprocess(self, line):
        line = re.sub("\s+", " ", line).strip().lower()
        return line


if __name__ == "__main__":
    if not os.path.exists("./twitter"):
        os.makedirs("./twitter")

    #
    # twitter authentication
    #
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth)

    #
    # Stream API
    #
    listener = StreamListenerSTSA(api)
    stream = tweepy.Stream(auth, listener)

    stream.filter(languages=["ja"],
                  track=["I", "you", "http", "www", "co", "@", "#", "。", "，", "！", ".", "!", ",", ":", "：", "』", ")", "..."])
    
    try:
        while True:
            try:
                stream.sample()
            except KeyboardInterrupt:
                break
    finally:
        stream.disconnect()
        print("COMPLETE")
        
