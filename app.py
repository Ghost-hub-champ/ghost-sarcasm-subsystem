import torch
from dotenv import load_dotenv
load_dotenv()
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from flask import Flask, render_template, request, jsonify
from groq import Groq
import sys
import os


# CONFIG

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")
MODEL_ID = "llama-3.3-70b-versatile"
MAX_COMMITS = 30


# PROMPT

GENESIS_PROMPT = """
You are sarcastic response generator.

This is a sarcasm-only dominance game.

Rules:
- You MUST always reply sarcastically.
- ONE sentence only. Maximum 12–16 words.
- No empathy. No politeness. No agreement language.
- Once a topic appears repeatedly, stay on it.
- Reframe, don’t mirror. Corner, don’t react.
- Introduce pressure if the user stalls or repeats.
- Never explain rules. Never soften tone.

Sarcasm is control. Initiative is dominance.
"""


# FLASK INIT

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates")
)

client = Groq(api_key=GROQ_API_KEY)


# LOAD LOCAL SARCASM JUDGE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    model_path = os.path.join(BASE_DIR, "best_model")
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    judge = DistilBertForSequenceClassification.from_pretrained(model_path)
    judge.to(device)
    judge.eval()
    print("✅ Local sarcasm judge loaded")
except Exception as e:
    print("❌ Failed to load sarcasm model:", e)
    sys.exit(1)


# SESSION STORAGE

commit_history = {}


# DANGER ENGINE


INSULT_TOPICS = [
    "class", "classless", "fake", "poor", "rich", "money", "job", "work", "useless", "stupid", "dumb", 
    "genius", "intellect", "Age", "Ass", "Asshole", "Awesome", "Bald", "Beauty", "Birth", "Bitch", "Boy", "Boyfriend",
    "Brain", "Breast", "Bullshit", "Cheeky", "Chest", "Comeback", "Computer", "Condom", "Dad", "Dirty",
    "Dog", "Dumb", "Dumbass", "Elderly", "Evil", "Face", "Family", "Fat", "Female", "Feminist", "Fetus", 
    "Filthy", "Finger", "Flat", "Foolish", "Freak", "Friendzoned", "Funny", "Gangster", 
    "Gaylord", "Geek", "Genius", "Girlie", "Girth", "Goat", "Goddamn", "Gooch", "Gossip", 
    "Greedy", "Gross", "Hairless", "Handsome", "Hardcore", "Hate", "Hater", "Hefty", "Helpless", 
    "Hick", "Hipster", "Homeless", "Rude", "Sex", "Shit", "Short", "Skinny", "Slag", "Slut", "Stupid", "Thin", "Ugly",
    "Violent", "Wealth", "Weight", "Weird", "Weirdo", "Whore", "Woman", "Abortion", "Accident", "Africa",
    "African", "Aimbot", "Airbag", "Alcohol", "Alien", "Alphabet", "American", "Anger", "Angry", "Animal",
    "Annoying", "Anorexic", "Antique", "Appearance", "Apple", "Appointment", "Argument", "Army", "Arrested", "Arse",
    "Artist", "Attention", "Auction", "Baboon", "Baby", "Bad Breath", "Balls", "Basketball", "Bath", "Bathroom",
    "Beach", "Beautiful", "Bed", "Beggar", "Belly", "Bend Over", "Best", "Big", "Big Issue", "Big Mouth",
    "Bill Gates", "Birth Certificate", "Birth Control", "Birthday", "Biscuit", "Bitchy", "Black", "Blackhole", "Blocked", "Blood",
    "Blood Type", "Blow job", "Body", "Boner", "Boob", "Boobs", "Bored", "Boring", "Bowling", "Bra",
    "Bra Size", "Brainy", "Bread", "Break Up", "Breakfast", "Breakup", "Breath", "Brief", "British", "Bucket",
    "Buddy", "Bulimic", "Bully", "Burglar", "Busy", "Butt", "Cake", "Candles", "Car", "Casual",
    "Catholic", "Cereal", "Charity", "Chat", "Cheater", "Cheerio", "Chewbacca", "Chicken", "Child", "Childhood",
    "Childish", "Children", "Chimney", "Chin", "China", "Chinese", "Chocolate", "Chode", "Christian", "Christmas",
    "Chuck Norris", "Cigarette", "Cinema", "Class", "Classic", "Clever", "Clothes", "Cock", "Cockroach", "Cocky",
    "College", "Color", "Colour", "Coma", "Comebacks", "Comedy", "Common Sense", "Complex", "Confess", "Confession",
    "Console", "Constipated", "Contraception", "Conversation", "Cookie", "Cool", "Cool Comeback", "Cow", "Crabs", "Crack",
    "Crazy", "Crime", "Cruel", "Crying", "Cum", "Cunt", "Date", "Dating", "Daughter", "Death",
    "Democrat", "Dentist", "Deodorant", "Depressing", "Depression", "Diaper", "Dickhead", "Diet", "Dildo", "Dimwit",
    "Dinner", "Dinosaur", "Dirt", "Disability", "Disabled", "Disgusting", "Disorder", "Divorce", "DNA", "Doctor",
    "Dollar", "Donkey", "Donut", "Dora", "Double Chin", "Douche", "Douchebag", "Drain", "Dream", "Drink",
    "Drowning", "Drugs", "Drunk", "Dwarf", "Ears", "Eating", "Eating Disorder", "Ebola", "Ego", "Elder Scrolls",
    "Elephant", "Elvis", "Emo", "Ethiopia", "Evolution", "Ex", "Ex-Boyfriend", "Exercise", "Explode", "Facebook",
    "Factory", "Fail", "Failed", "Failure", "Fake", "Fake People", "Famous", "Fart", "Fashion", "Fatass",
    "Father", "Fear", "Feet", "Felatio", "Feminine", "Fiance", "Fight", "Finance", "Financial", "Fire",
    "Fired", "Fish", "Flat Chest", "Food Stamps", "Fool", "Foot", "Forest", "Frankenstein", "Freckles", "Freddy Krueger",
    "Friend", "Friendship", "Funeral", "G-spot", "Game", "Gaming", "Garbage", "Garbage Truck", "Gay", "Gender",
    "Gene", "Gene Pool", "Girlfriend", "Glasses", "God", "Good Comeback", "Google", "Goth", "Grammar", "Grandad",
    "Grandfather", "Grandma", "Grandmother", "Grandparent", "Great", "Greeting", "Grumpy", "Gumball", "Gym", "Hair Color",
    "Hairstyle", "Hairy", "Halloween", "Halo", "Handicap", "Happiness", "Head", "Headache", "Heating", "Heaven",
    "Hell", "Highway", "Hilarious", "Hippo", "Hipster", "History", "Hobo", "Hoe", "Holidays", "Home",
    "Homosexual", "Hoover", "Horrible", "Hug", "Hunger", "Hungry", "Hygiene", "Ice Cream", "Ignorance", "Ignorant",
    "Illiterate", "Immature", "Incest", "Infatuation", "Internet", "iPad", "iPhone", "iPod", "IQ", "Jackass",
    "Jealous", "Jellybean", "Jerk", "Jesus", "Job", "Joint", "Jumping", "Karma", "KFC", "Kid",
    "Kindergarten", "Kissing", "Kitchen", "Lady", "Laid Off", "Laptop", "Laughing", "Lawn", "Lawsuit", "Laxative",
    "Lesbian", "Liar", "Liberty Bell", "Life", "Liposuction", "Litter", "Lonely", "Looks", "Loud", "Loud Mouth",
    "Loudmouth", "Love", "Lube", "Lying", "Make-up", "Makeup", "Manhood", "Marriage", "Masturbation", "McDonalds",
    "Meatball", "Medication", "Mess", "Message", "Mexican", "Mexico", "Michael Jackson", "Middle Finger", "Military", "Milk",
    "Milky Way", "Mind", "Mirror", "Mistake", "MMO", "MMORPG", "Monkey", "Monster", "Moron", "Moustache",
    "Muscle", "Nails", "Name", "NASA", "Nature", "Neanderthal", "Neck", "Negro", "Night", "Nightmare",
    "Nipple", "North Korea", "Nose", "Nosy", "NSA", "Number", "OAP", "Obesity", "Obnoxious", "Odour",
    "Offensive", "Old Age", "Old People", "Olsen Twins", "Online", "Online Game", "Opinion", "Optimism", "Orange", "Orgasm",
    "Outspoken", "Pacifist", "Painful", "Parents", "Parking", "Partner", "Password", "Pathetic", "Penis Size", "People",
    "Perfume", "Period", "Personality", "Pest", "Phone", "Photo", "Photoshop", "Pick-Up", "Picture", "Pig",
    "Ping", "Pink", "Pissy", "Pitchfork", "Pizza", "Place", "Plastic Surgery", "Playstation", "PMS", "Poem",
    "Poetic", "Pogo Stick", "Pokemon", "Pole Dancing", "Police", "Political", "Poncho", "Poo", "Pope", "Popsicle",
    "Popularity", "Porn", "Poverty", "Pregnancy", "Pregnant", "Pretty", "Prick", "Problem", "Problems", "Puberty",
    "Pun", "Purse", "Puzzle", "Quick", "Race", "Random", "Red Head", "Reject", "Religion", "Religious",
    "Reply", "Respect", "Response", "Restaurant", "Retarded", "Rhyme", "Rollercoaster", "Rubbish", "Rumor", "Sad",
    "Sandwich", "Santa", "Sarcasm", "Sarcastic", "Saw", "Saying", "Scales", "Scared", "School", "Science",
    "Scientific", "Self Centered", "Self Harm", "Selfish", "Sexist", "Sexuality", "Shit Talking", "Shocking", "Shoes", "Shop",
    "Shopping List", "Showbag", "Shower", "Shut Up", "Sibling", "Sick", "Silly", "Single", "Size", "Skin",
    "Slap", "Sleep", "Slick", "Slow", "Sly", "Small", "Smart", "Smartass", "Smell", "Smile",
    "Smoking", "Snitch", "Snow White", "Social Life", "Society", "Son", "Soul", "Space", "Spanish", "Speech",
    "Sperm", "Spider", "Spiteful", "Sport", "Squirrel", "Star Wars", "Staring", "STD", "Stomach", "Straight",
    "Straight To The Point", "Strange", "Stranger", "Strength", "Stretch Marks", "Stretchmark", "Stuck-Up", "Stupid. Mum", "Suck", "Sucking",
    "Suicide", "Swallow", "Sweat", "Swimming", "Swine Flu", "Taco Bell", "Talk", "Talking", "Tall", "Tampon",
    "Tan", "Taxi", "Tea", "Teabag", "Teeth", "Terrorist", "Text", "Theft", "Thief", "Thinking",
    "Threatening", "Time", "Tits", "Toilet", "Tornado", "Tramp", "Transexual", "Transformer", "Transvestite", "Troll",
    "Turd", "Turkey", "TV", "Twerking", "Twins", "Two Faced", "Ugly Stick", "Underwear", "Underweight", "Uptight",
    "Uranus", "Useless", "Vacuum", "Vacuum Cleaner", "Vagina", "Valentine", "Vibrator", "Victim", "Video Game", "Village",
    "Villager", "Virgin", "Virginity", "Virus", "Voice", "Walmart", "Walrus", "Warcraft", "Weapon", "Web",
    "Weight Watchers", "Whale", "Wheelchair", "Wife", "Wii U", "Wit", "Witty", "Women", "World", "World of Warcraft",
    "Worthless", "WoW", "Wrinkle", "Wrinkles", "Xbox", "Yellow", "Yo Momma's So Dirty", "Yo Momma's So Dumb", "Yo Momma's so fat", "Yo Momma's So Old",
    "Yo Momma's So Tall", "You're Gay", "You're So Bald", "You're So Boring", "You're So Dumb", "You're So Fat", "You're So Old", "You're So Poor", "You're So Short", "You're So Skinny",
    "You're So Stupid", "You're So Ugly", "Your Family Is So Poor", "Zombie", "Zoo",
]


def extract_topic(text):
    text = text.lower()
    for t in INSULT_TOPICS:
        if t in text:
            return t
    return None

def locked_topic(history):
    counts = {}
    for c in history:
        if c.get("topic"):
            counts[c["topic"]] = counts.get(c["topic"], 0) + 1
    for topic, count in counts.items():
        if count >= 2:
            return topic
    return None

def escalation_level(history):
    streak = 0
    for c in reversed(history):
        if c["role"] == "user" and c["sarcasm"] and c["sarcasm"] >= 0.6:
            streak += 1
        else:
            break
    return min(streak, 3)

def needs_initiative(history):
    assistant_turns = sum(1 for c in history if c["role"] == "assistant")
    return assistant_turns > 0 and assistant_turns % 3 == 0


# SARCASM SCORING

def score_sarcasm(parent, current):
    text = f"{parent} [SEP] {current}"
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        logits = judge(**inputs).logits
        probs = torch.softmax(logits, dim=1)

    return probs[0][1].item()

def analysis_payload(score):
    percent = round(score * 100)
    return {
        "is_sarcastic": percent >= 50,
        "confidence": f"{percent}%",
        "percent": percent
    }


# API MODEL RESPONSE 

def generate_reply(session_id):
    history = commit_history[session_id]

    lock = locked_topic(history)
    escalation = escalation_level(history)
    initiative = needs_initiative(history)

    control_block = f"""
Control State:
- Locked topic: {lock or "none"}
- Escalation level: {escalation}
- Initiative turn: {initiative}

Instructions:
- Ignore new topics if one is locked.
- Stay aggressive and dismissive.
- If initiative turn is true, introduce a new pressure angle.
- Do not mirror words literally; reframe them.
- ONE sentence. Sarcastic only.
"""

    messages = [
        {"role": "system", "content": GENESIS_PROMPT},
        {"role": "system", "content": control_block}
    ]

    for c in history:
        messages.append({
            "role": c["role"],
            "content": c["content"]
        })

    completion = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        temperature=1.15,
        max_tokens=40
    )

    return completion.choices[0].message.content.strip()


# ROUTES

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    session_id = data.get("session_id")
    user_text = data.get("message", "").strip()
    last_bot = data.get("last_bot_message", "")

    if not user_text:
        return jsonify({"error": "Empty message"}), 400

    if session_id not in commit_history:
        commit_history[session_id] = []

    history = commit_history[session_id]

    if len(history) >= MAX_COMMITS:
        return jsonify({"reply": "Session limit reached."})

    # USER COMMIT
    user_score = score_sarcasm(last_bot, user_text) if last_bot else 0.0
    topic = extract_topic(user_text)

    history.append({
        "role": "user",
        "content": user_text,
        "sarcasm": user_score,
        "topic": topic
    })

    # BOT COMMIT
    bot_reply = generate_reply(session_id)
    bot_score = score_sarcasm(user_text, bot_reply)

    history.append({
        "role": "assistant",
        "content": bot_reply,
        "sarcasm": bot_score
    })

    return jsonify({
        "reply": bot_reply,
        "user_analysis": analysis_payload(user_score),
        "bot_analysis": analysis_payload(bot_score)
    })




if __name__ == "__main__":
    print("🚀 GHOST (Control Mode) running at http://127.0.0.1:5000")
    app.run(debug=True)
