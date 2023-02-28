import streamlit as st
import pickle
import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from streamlit_ace import st_ace

le = LabelEncoder()
data = pd.read_csv("train1.csv",encoding="utf-8")
y = data["hate speech"]

    # label encoding
y = le.fit_transform(y)


# Load the trained machine learning model##
model = pickle.load(open("model.pkl", "rb"))
cv = pickle.load(open("transform.pkl", "rb"))

def bdtext_cleaner(text):
    whitespace= re.compile(u"[\s\u0020\u00a0\u1680\u180e\u202f\u205f\u3000\u2000-\u200a]+", re.UNICODE)
    bangla_fullstop = u"\u0964"
    punctSeq   = u"['\"“”‘’]+|[.?!,…]+|[:;]+"
    punc = u"[(),$%^&*$#@+={}\[\]:\"|\'\~`<>/,¦!?½£¶¼©⅐⅑⅒⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞⅟↉¤¿º;-]+"
    eng_char  =  u"[abcdefghijklmnopqrstwvxyz,ABCDEFGHIJKLMNOPQRSTWVXYZ]+"
    text= whitespace.sub(" ",text).strip()
    text= re.sub(punctSeq, "", text)
    text= re.sub(bangla_fullstop, " ",text)
    text= re.sub(punc, "", text)
    text= re.sub(eng_char, "", text)
    return text

# Define the correct username and password
username = 'user'
password = 'password'

# Define the input form using the st_ace function
login_code = st_ace(
    placeholder='Enter your password',
    language='python',
    keybinding='vscode',
    show_gutter=False,
    font_size=18,
    theme='chrome',
    height=150
)
# Define a submit button to check the password
def main():
    if st.button('Login'):
        if login_code == password:
            st.success('You have successfully logged in!')
            st.write('')
            st.write('')
            st.header("পোস্ট বক্স")
            # Create a text input for the post content
            post_content = st.text_area("পোস্টের বিস্তারিত:")
            # Create a button to submit the post
            if st.button("পোস্ট করুন"):
                # Save the post, title, and name to a file or database
                with open("posts.txt", "a", encoding="utf-8") as f:
                    # f.write(f"{name}\n")
                    f.write(f"{post_content}\n\n")
                st.success("পোস্ট সফলভাবে পোস্ট হয়েছে!")

            # Display existing posts
            st.write("বিদ্যমান পোস্টসমূহ:")
            with open("posts.txt", "r", encoding="utf-8") as f:
                posts = f.readlines()
            for post in posts:
                st.write(post)

            st.title("মন্তব্য বক্স")

            # Create a text input for the commenter's name
            commentor_name = st.text_input("নাম:")

            # Create a text input for the comment
            comment = st.text_area("মন্তব্য করুন:")

            # Create a button to submit the comment
            if st.button("জমা দিন"):
                # Predict whether the comment should be filtered or not
                comment = bdtext_cleaner(comment)
                dat = [comment]
                # creating the vector
                vect = cv.transform(dat).toarray()
                # prediction
                my_pred = model.predict(vect)
                my_pred = le.inverse_transform(my_pred)
                result = my_pred[0]
                if result == 0:
                    comment = comment
                else:
                    comment = "  "
                # Save the comment and name to a file or database
                with open("comments.txt", "a", encoding="utf-8") as f:
                    f.write(f"{commentor_name}: {comment}\n")
                st.success("মন্তব্য জমা দেওয়া হয়েছে!")

            # Display existing comments
            st.write("বিদ্যমান মন্তব্যসমূহ:")
            with open("comments.txt", "r", encoding="utf-8") as f:
                comments = f.readlines()
            for comment in comments:
                st.write(comment)
        else:
            st.error('Invalid password. Please try again.')


# def main():
#     st.set_page_config(page_title="মন্তব্য বক্স", page_icon=":speech_balloon:")
#     # # Create a photo section
#     # st.header("ফটো সেকশন")
#     # uploaded_files = st.file_uploader("এখানে আপনি ফটো আপলোড করতে পারেন:", accept_multiple_files=True)
#     # if uploaded_files:
#     #     st.success(f"মোট {len(uploaded_files)} টি ফাইল সফলভাবে আপলোড হয়েছে:")
#     #     for uploaded_file in uploaded_files:
#     #         st.image(uploaded_file, use_column_width=True, caption=uploaded_file.name)
#
#     # Create a text input for the commenter's name
#     # name = st.text_input("আপনার নাম:")
#     st.header("পোস্ট বক্স")
#     # Create a text input for the post content
#     post_content = st.text_area("পোস্টের বিস্তারিত:")
#     # Create a button to submit the post
#     if st.button("পোস্ট করুন"):
#         # Save the post, title, and name to a file or database
#         with open("posts.txt", "a", encoding="utf-8") as f:
#             # f.write(f"{name}\n")
#             f.write(f"{post_content}\n\n")
#         st.success("পোস্ট সফলভাবে পোস্ট হয়েছে!")
#
#     # Display existing posts
#     st.write("বিদ্যমান পোস্টসমূহ:")
#     with open("posts.txt", "r", encoding="utf-8") as f:
#         posts = f.readlines()
#     for post in posts:
#         st.write(post)
#
#     st.title("মন্তব্য বক্স")
#
#     # Create a text input for the commenter's name
#     commentor_name = st.text_input("নাম:")
#
#     # Create a text input for the comment
#     comment = st.text_area("মন্তব্য করুন:")
#
#     # Create a button to submit the comment
#     if st.button("জমা দিন"):
#         # Predict whether the comment should be filtered or not
#         comment = bdtext_cleaner(comment)
#         dat = [comment]
#         # creating the vector
#         vect = cv.transform(dat).toarray()
#         # prediction
#         my_pred = model.predict(vect)
#         my_pred = le.inverse_transform(my_pred)
#         result = my_pred[0]
#         if result == 0:
#             comment = comment
#         else:
#             comment = "  "
#         # Save the comment and name to a file or database
#         with open("comments.txt", "a", encoding="utf-8") as f:
#             f.write(f"{commentor_name}: {comment}\n")
#         st.success("মন্তব্য জমা দেওয়া হয়েছে!")
#
#     # Display existing comments
#     st.write("বিদ্যমান মন্তব্যসমূহ:")
#     with open("comments.txt", "r", encoding="utf-8") as f:
#         comments = f.readlines()
#     for comment in comments:
#         st.write(comment)
#



if __name__ == "__main__":
    main()
