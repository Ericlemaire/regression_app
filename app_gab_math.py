import streamlit as st
import random

st.title('Jeu d\'Opérations Mathématiques')

# Initialiser les variables de session
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'question' not in st.session_state:
    st.session_state.question = ('multiplication', random.randint(1, 12), random.randint(1, 12))
if 'operation' not in st.session_state:
    st.session_state.operation = 'multiplication'

# Générer une nouvelle question
def new_question(operation):
    num1 = random.randint(1, 12)
    num2 = random.randint(1, 12)
    st.session_state.question = (operation, num1, num2)

# Réinitialiser le jeu
def reset_game():
    st.session_state.score = 0
    new_question(st.session_state.operation)

# Sélection de l'opération
operation = st.selectbox('Choisissez le type d\'opération', ('multiplication', 'addition', 'soustraction'))
st.session_state.operation = operation

# Générer une nouvelle question si le type d'opération change
if st.session_state.operation != operation:
    new_question(operation)

# Obtenir les éléments de la question actuelle
op, num1, num2 = st.session_state.question

if op == 'multiplication':
    question_text = f'Combien font {num1} x {num2} ?'
    correct_answer = num1 * num2
elif op == 'addition':
    question_text = f'Combien font {num1} + {num2} ?'
    correct_answer = num1 + num2
else:  # soustraction
    question_text = f'Combien font {num1} - {num2} ?'
    correct_answer = num1 - num2

answer = st.number_input(question_text, min_value=-144, step=1, key='answer')

if st.button('Soumettre'):
    if answer == correct_answer:
        st.success('Bonne réponse!')
        st.session_state.score += 1
        new_question(st.session_state.operation)
    else:
        st.error('Mauvaise réponse. Réessayez!')
        new_question(st.session_state.operation)

st.write(f'Votre score: {st.session_state.score}')

# Afficher des ballons lorsque le score atteint 20
if st.session_state.score > 0 and st.session_state.score % 20 == 0:
    st.balloons()
    st.session_state.score = 0  # Réinitialiser le score après avoir montré les ballons

# Ajouter un bouton pour recommencer une nouvelle partie
if st.button('Recommencer une nouvelle partie'):
    reset_game()
    st.experimental_rerun()