from functools import wraps
from flask import Flask, redirect, url_for, session, request, flash
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os

# Database setup
def init_db():
    if not os.path.exists('users.db'):
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE users
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      username TEXT UNIQUE NOT NULL,
                      password TEXT NOT NULL,
                      role TEXT NOT NULL DEFAULT 'user')''')

        # Add admin user if not exists
        admin_password = generate_password_hash('admin123')
        c.execute("INSERT OR IGNORE INTO users (username, password, role) VALUES (?, ?, ?)",
                  ('admin', admin_password, 'admin'))
        conn.commit()
        conn.close()


init_db()


# Authentication decorators
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)

    return decorated_function


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login', next=request.url))
        if session.get('role') != 'admin':
            flash('You need admin privileges to access this page')
            return redirect(url_for('index'))
        return f(*args, **kwargs)

    return decorated_function