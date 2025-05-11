from flask import Blueprint, redirect, url_for, request, session, render_template
from authlib.integrations.flask_client import OAuth
from flask_login import login_user, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User

auth = Blueprint("auth", __name__)
oauth = OAuth()

# 🔹 Initialize OAuth with the Flask app
def init_oauth(app):
    oauth.init_app(app)

    oauth.register(
        "google",
        client_id=app.config["GOOGLE_CLIENT_ID"],  # ✅ Fetch from app.config
        client_secret=app.config["GOOGLE_CLIENT_SECRET"],  # ✅ Fetch from app.config
        authorize_url="https://accounts.google.com/o/oauth2/auth",
        access_token_url="https://oauth2.googleapis.com/token",
        client_kwargs={"scope": "openid email profile"},
        jwks_uri="https://www.googleapis.com/oauth2/v3/certs",  # Add this line
    )

# 🔹 Google Login Route
@auth.route("/login/google")
def login_google():
    return oauth.google.authorize_redirect(url_for("auth.google_authorized", _external=True))

# 🔹 Google OAuth Callback
@auth.route("/login/google/authorized")
def google_authorized():
    token = oauth.google.authorize_access_token()
    if not token:
        return "Authentication failed. Please try again."

    user_info = oauth.google.get("https://www.googleapis.com/oauth2/v2/userinfo").json()
    email = user_info.get("email")

    if not email:
        return "Failed to fetch email from Google."

    # 🔹 Check if user exists in DB
    user = User.query.filter_by(email=email).first()
    if not user:
        user = User(email=email, google_id=user_info["id"])
        db.session.add(user)
        db.session.commit()

    login_user(user)  # 🔹 Log the user in
    return redirect(url_for("dashboard"))

# 🔹 Manual Registration Route
@auth.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        hashed_password = generate_password_hash(password, method="pbkdf2:sha256")

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return "User already exists! Try logging in."

        new_user = User(email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for("auth.login"))

    return render_template("register.html")

# 🔹 Manual Login Route
@auth.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for("dashboard"))

    return render_template("login.html")

# 🔹 Logout Route
@auth.route("/logout")
def logout():
    logout_user()
    session.pop("google_token", None)
    return redirect(url_for("auth.login"))
