# reviews/forms.py
from django import forms

class ReviewForm(forms.Form):
    review_text = forms.CharField(label='Ваш отзыв', max_length=1000, widget=forms.Textarea)
