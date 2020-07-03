import os
from flask import Blueprint, current_app
from flask_bootstrap import Bootstrap


from app.controllers.users_controller import index as users_index
from app.controllers.users_controller import tweets as users_tweets
from app.controllers.users_controller import respond as users_respond



template_dir = os.path.abspath('app/views/users')

user_blueprints = Blueprint('users', 'api', template_folder=template_dir)
user_blueprints.add_url_rule('/', view_func=users_index, methods=['GET', 'POST'])
user_blueprints.add_url_rule('/tweets', view_func=users_tweets, methods=['GET', 'POST'])
user_blueprints.add_url_rule('/respond', view_func=users_respond, methods=['GET', 'POST'])



