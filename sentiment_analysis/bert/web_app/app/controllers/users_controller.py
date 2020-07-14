from flask import request, jsonify, render_template, url_for, request, redirect
import app.helpers.user_service as us
import app.helpers.twitter as tweet
def index():
  if request.method == "POST":
    review = request.form['comment']
    result = us.infer(review)
    result = {'sentiment': result, 'text': review}
    return render_template('show.html', result=result)

  return render_template('index.html')


def respond():
  fdata = request.form
  return render_template('respond.html')


def tweets():
  data = tweet.fetch()
  data = us.analyse(data)
  return render_template('tweets.html', result=data)

