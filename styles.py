css = '''
<style>
html {
    box-sizing: border-box;
    margin:0 auto;
    padding: 0;
    background-color: #000;
}
.stApp {
    color: #fff;
}
.stApp header {
    background-color: #000;
}
.stDeployButton button {
    background-color: #fff !important;
    color: #000;
}
section.main {
    background-color: #000;
}

.stBottom {
background-color: #475063;
}

h1 {
    color: #bebebe;
    padding-bottom: 1.5em;
}

.st-emotion-cache-4oy321 {
    background-color: #475063;
    color: #fff !important;
}
.st-emotion-cache-4oy321 p {
    margin: 0 1.2em 1rem 1.2em;
    color: #fff;
}
.st-emotion-cache-4oy321 li {
    color: #fff;
}
.st-emotion-cache-uzeiqp {
    color: #000;
}
.st-emotion-cache-1bpjhwx label p,  #f103f5e7 {
    color: #fff;
}
.st-emotion-cache-vxumw0 {
    color: #000;
}
.stButton button {
    color: #000;
}

.stFileUploaderFileName {
    color: #000;
}

[data-testid="stBottom"] .st-emotion-cache-uhkwx6{
    border-top: 2px solid #fff;
    background-color: #000;
    display: flex;
    justify-content: center;
    align-items: center;
}
.st-emotion-cache-arzcut {
    padding: 0 !important;
    margin: 2em 0;
}

.chat-message {
    display: flex;
    align-items: center;
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
}

.chat-message.user {
    background-color: #2b313e;
}
.chat-message.bot {
    background-color: #475063;
}
.chat-message .avatar {
  width: 10%;
}
.chat-message .avatar img {
  max-width: 68px;
  max-height: 68px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/5rrMjP3/images.jpg">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

ai_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/vqvQHNh/openai-chatgpt-logo-icon-free-png.webp" style="max-height: 68px; max-width: 68px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''
