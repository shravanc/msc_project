from flask import request, jsonify, render_template, url_for, request, redirect
import app.helpers.user_service as us

def index():
  if request.method == "POST":
    review = request.form['comment']
    result = us.infer(review)
    return render_template('show.html', result=result)

  return render_template('index.html')


def respond():
  fdata = request.form 
  return render_template('respond.html')
