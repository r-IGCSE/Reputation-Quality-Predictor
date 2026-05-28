from mongoengine import EmbeddedDocument, EmbeddedDocumentField, connect, Document, StringField, DateTimeField, IntField, ListField, ReferenceField
import os
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("MONGO_URL"):
    raise Exception("MONGO_URL environment variable is not set")

connect(host=os.getenv("MONGO_URL"))

class Message(EmbeddedDocument):
    author = StringField(required=True)
    content = StringField(required=True)
    attachmentUrls = ListField(StringField(), default=[])
    
class ReputationQueue(Document):
    input = ListField(EmbeddedDocumentField(Message), required=True)
    reputationDataObjectId = StringField(required=True)
    __v = IntField(db_field='__v') 
    
    meta = {'collection': 'reputationqueues'}
    
class ReputationData(Document):
    reppedUser = StringField(required=True)
    reppedBy = StringField(required=False)
    repNumber = IntField(required=True)
    when = DateTimeField(required=True)
    channelId = StringField(required=True)
    guildId = StringField(required=True)
    __v = IntField(db_field='__v') 
    
    meta = {'collection': 'reputationdatas'}
