import firebase_admin
from firebase_admin import credentials, firestore, db
import pandas as pd

# Use a service account
cred = credentials.Certificate("plant2iot-firebase-adminsdk-5wx4h-9c5374ca50.json")

firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://plant2iot-default-rtdb.firebaseio.com/'})

db2 = firestore.client()

obj1 = {
  "Name": "Aniruddha Jana",
  "Age": 21,
  "Purpose": ["Learning", "Fun", "Work"]
}

obj2 = {
  "Name": "Prerana Jana",
  "Age": 21,
  "Purpose": ["Learning", "Fun", "Work"]
}

data = [obj1, obj2]

'''
### CRUD Operations ###
### Uncomment the code to perform the operation ###
### Comment the code after performing the operation ###
### FireStore ###
# Addong new data
# for i in data:
#   doc_ref = db2.collection(u'users').document(i["Name"])
#   doc_ref.set(i)
  
# Updating data
# up_ref = db2.collection(u'users').document("Prerana Jana")
# up_ref.update({"Age": 20})
# # setting timestamp
# up_ref.update({
#     u'timestamp': firestore.SERVER_TIMESTAMP
# })

# Getting data
# doc_ref = db2.collection(u'users').document("Aniruddha Jana")
# doc = doc_ref.get()
# if doc.exists:
#   print(doc.to_dict())
# else:
#   print("No such document!")

# # Deleting data
# db2.collection(u'users').document("Anushmita Jana").delete()


### Realtime Database ###
'''
'''
# Adding new data
# ref.child('users').set(data)    # Indexing starts from 0

# for i in data:
#   ref.child('users').child(i["Name"]).set(i)  # Indexing starts with the name

# Updating data
# ref.child('users').child("Aniruddha Jana").update({"Age": 22})
# Multiple updates
# ref.child('users').child("1").update({
#   "Age": 22,
#   "Name": "Prerana Kuila"
# })
'''
#Getting data
ref_iot = db.reference('/iot/')
ds = ref_iot.get()
#print(ds)
ref_count = db.reference('/count/')
ds2 = ref_count.get()
print("Count:",ds2)

df = pd.DataFrame(ds)
print(df.head())

print("Shape:",df.shape)