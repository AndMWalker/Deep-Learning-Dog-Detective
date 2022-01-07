from telegram.ext import *
from io import BytesIO
import cv2
import numpy as np
from tensorflow.keras.models import load_model

with open('API_Token.txt', 'r') as f: #reading Telegram API
    TOKEN = f.read()

class_names = ['Chihuahua', 'Maltese dog', 'Shih-Tzu', 'Beagle', 'Border terrier', 'Yorkshire terrier', 'Golden retriever',
               'Labrador retriever', 'German short-haired pointer', 'Cocker spaniel', 'Border collie', 'Rottweiler', 'German shepherd',
               'French bulldog', 'Great Dane', 'Malamute', 'Siberian_husky', 'Pug', 'Pomeranian', 'Chow', 'Pembroke', 'Poodle'] #list of possible breeds for deduction

def start(update, context):
    update.message.reply_text("Greetings! I am Dog Detective Bot.\nI can sniff out different breeds of dogs if you show me an image.\n/help for more info")

def help(update, context):
    update.message.reply_text("""
    Send me an image of a dog and I will try to deduce it's breed.\n\nPlease take note that only 22 breeds of dogs are able to be detected at the moment.\n\n/list for the full list of dog breeds.\n\nFor more accurate results, please send photos that are high quality and show the face and body of the dog clearly.
    """)

def list(update, context):
    update.message.reply_text("""
    Chihuahua, Maltese dog, Shih-Tzu, Beagle, Border terrier, Yorkshire terrier, Golden retriever, Labrador retriever, German short-haired pointer, Cocker spaniel, Border collie, Rottweiler, German shepherd, French bulldog, Great Dane, Malamute, Siberian husky, Pug, Pomeranian, Chow, Pembroke, Poodle
    """)

def model_loader():
    global model
    model = load_model('deep_learning_dogs.h5')
    print('Model loaded')
#loading pre-trained VGG16 model

def handle_photo(update, context):
    file = context.bot.get_file(update.message.photo[-1].file_id)
    f = BytesIO(file.download_as_bytearray())
    files_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)

    img = cv2.imdecode(files_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (224,224), interpolation=cv2.INTER_AREA)

    prediction = model.predict(np.array([img]))
    update.message.reply_text(f"This looks like a {class_names[np.argmax(prediction)]}")
#reads photos sent to the bot and evaluates the images in the model

def main():
    model_loader()
    updater = Updater(TOKEN, use_context=True)
    disp = updater.dispatcher
    disp.add_handler(CommandHandler("start", start))
    disp.add_handler(CommandHandler("help", help))
    disp.add_handler(CommandHandler("list", list))
    disp.add_handler(MessageHandler(Filters.photo, handle_photo))
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()




