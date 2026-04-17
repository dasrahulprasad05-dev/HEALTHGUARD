"""
HealthGuard DB — app.py
Complete Flask application with:
  - User registration and login
  - Patient dashboard with prediction history
  - Doctor dashboard with all patients
  - ML prediction saving to database
  - PDF report generation
  - Alert system for high risk
"""
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt
from config import Config
from models import db, User, Prediction, Alert
from predictor import predict_all
from report import generate_report
import json, io
from datetime import datetime

# ── APP SETUP ─────────────────────────────────────
app = Flask(__name__)
app.config.from_object(Config)

db.init_app(app)
bcrypt = Bcrypt(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please login to access this page.'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ── CREATE TABLES ─────────────────────────────────
with app.app_context():
    db.create_all()
    # Create a default doctor account if not exists
    if not User.query.filter_by(email='doctor@healthguard.com').first():
        hashed = bcrypt.generate_password_hash('doctor123').decode('utf-8')
        doctor = User(name='Dr. Health Guard', email='doctor@healthguard.com',
                      password=hashed, role='doctor')
        db.session.add(doctor)
        db.session.commit()
        print("Default doctor account created: doctor@healthguard.com / doctor123")

# ══════════════════════════════════════════════════
# AUTH ROUTES
# ══════════════════════════════════════════════════

@app.route('/')
def index():
    if current_user.is_authenticated:
        if current_user.role == 'doctor':
            return redirect(url_for('doctor_dashboard'))
        return redirect(url_for('patient_dashboard'))
    return render_template('index.html')

@app.route('/register', methods=['GET','POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        name       = request.form.get('name')
        email      = request.form.get('email')
        password   = request.form.get('password')
        age        = request.form.get('age')
        sex        = request.form.get('sex')
        mobile     = request.form.get('mobile')
        blood_group= request.form.get('blood_group')

        # Check email already exists
        if User.query.filter_by(email=email).first():
            flash('Email already registered. Please login.', 'danger')
            return redirect(url_for('register'))

        hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(name=name, email=email, password=hashed_pw,
                    age=age, sex=sex, mobile=mobile,
                    blood_group=blood_group, role='patient')
        db.session.add(user)
        db.session.commit()
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        email    = request.form.get('email')
        password = request.form.get('password')
        user     = User.query.filter_by(email=email).first()

        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            flash(f'Welcome back, {user.name}!', 'success')
            if user.role == 'doctor':
                return redirect(url_for('doctor_dashboard'))
            return redirect(url_for('patient_dashboard'))
        else:
            flash('Invalid email or password.', 'danger')

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('login'))

# ══════════════════════════════════════════════════
# PATIENT ROUTES
# ══════════════════════════════════════════════════

@app.route('/dashboard')
@login_required
def patient_dashboard():
    if current_user.role == 'doctor':
        return redirect(url_for('doctor_dashboard'))

    # Get last 10 predictions for this patient
    predictions = Prediction.query.filter_by(user_id=current_user.id)\
                  .order_by(Prediction.created_at.desc()).limit(10).all()

    # Get unread alerts
    alerts = Alert.query.filter_by(user_id=current_user.id, is_read=False)\
             .order_by(Alert.created_at.desc()).all()

    # Chart data — last 6 predictions for trend graph
    chart_preds = Prediction.query.filter_by(user_id=current_user.id)\
                  .order_by(Prediction.created_at.asc()).limit(6).all()

    chart_labels  = [p.created_at.strftime('%d %b') for p in chart_preds]
    chart_heart   = [p.heart_risk    for p in chart_preds]
    chart_diabetes= [p.diabetes_risk for p in chart_preds]
    chart_kidney  = [p.kidney_risk   for p in chart_preds]

    # Latest prediction for summary cards
    latest = predictions[0] if predictions else None

    return render_template('patient_dashboard.html',
        predictions=predictions,
        alerts=alerts,
        latest=latest,
        chart_labels=json.dumps(chart_labels),
        chart_heart=json.dumps(chart_heart),
        chart_diabetes=json.dumps(chart_diabetes),
        chart_kidney=json.dumps(chart_kidney),
        alert_count=len(alerts),
    )

@app.route('/predict', methods=['GET','POST'])
@login_required
def predict():
    if request.method == 'POST':
        data = request.form.to_dict()

        # Run ML prediction
        result = predict_all(data)

        # ── Save to database ──────────────────────
        pred = Prediction(
            user_id       = current_user.id,
            age           = float(data.get('age', 0)),
            sex           = float(data.get('sex', 0)),
            cp            = float(data.get('cp', 0)),
            trestbps      = float(data.get('trestbps', 0)),
            chol          = float(data.get('chol', 0)),
            fasting_sugar = float(data.get('fasting_sugar', 0)),
            glucose       = float(data.get('glucose', 0)),
            thalach       = float(data.get('thalach', 0)),
            exang         = float(data.get('exang', 0)),
            oldpeak       = float(data.get('oldpeak', 1.0)),
            bmi           = float(data.get('bmi', 0)),
            pregnancies   = float(data.get('pregnancies', 0)),
            insulin       = float(data.get('insulin', 125)),
            hemo          = float(data.get('hemo', 0)),
            sc            = float(data.get('sc', 0)),
            heart_risk    = result['heart']['probability'],
            diabetes_risk = result['diabetes']['probability'],
            kidney_risk   = result['kidney']['probability'],
            overall_score = result['overall']['score'],
            heart_level   = result['heart']['risk_level'],
            diabetes_level= result['diabetes']['risk_level'],
            kidney_level  = result['kidney']['risk_level'],
            overall_status= result['overall']['status'],
        )
        db.session.add(pred)
        db.session.commit()

        # ── Check for alerts (risk increased by 15%+) ──
        prev = Prediction.query.filter_by(user_id=current_user.id)\
               .order_by(Prediction.created_at.desc()).offset(1).first()

        if prev:
            for disease, new_risk, old_risk in [
                ('heart',    result['heart']['probability'],    prev.heart_risk),
                ('diabetes', result['diabetes']['probability'], prev.diabetes_risk),
                ('kidney',   result['kidney']['probability'],   prev.kidney_risk),
            ]:
                if new_risk - old_risk >= 15:
                    alert = Alert(
                        user_id  = current_user.id,
                        disease  = disease,
                        old_risk = old_risk,
                        new_risk = new_risk,
                        message  = f'Your {disease.title()} risk increased from {old_risk:.1f}% to {new_risk:.1f}%! Please consult a doctor.'
                    )
                    db.session.add(alert)
            db.session.commit()

        return render_template('result.html',
            result=result, pred=pred,
            patient=current_user)

    return render_template('predict.html', patient=current_user)

@app.route('/history')
@login_required
def history():
    if current_user.role == 'doctor':
        return redirect(url_for('doctor_dashboard'))
    predictions = Prediction.query.filter_by(user_id=current_user.id)\
                  .order_by(Prediction.created_at.desc()).all()
    return render_template('history.html', predictions=predictions)

@app.route('/download_report/<int:pred_id>')
@login_required
def download_report(pred_id):
    pred = Prediction.query.get_or_404(pred_id)
    # Security — patient can only download own reports
    if pred.user_id != current_user.id and current_user.role != 'doctor':
        flash('Access denied.', 'danger')
        return redirect(url_for('patient_dashboard'))

    patient = User.query.get(pred.user_id)
    pdf_bytes = generate_report(pred, patient)
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f'HealthGuard_Report_{patient.name}_{pred.created_at.strftime("%d%m%Y")}.pdf'
    )

@app.route('/mark_alert_read/<int:alert_id>')
@login_required
def mark_alert_read(alert_id):
    alert = Alert.query.get_or_404(alert_id)
    if alert.user_id == current_user.id:
        alert.is_read = True
        db.session.commit()
    return redirect(url_for('patient_dashboard'))

# ══════════════════════════════════════════════════
# DOCTOR ROUTES
# ══════════════════════════════════════════════════

@app.route('/doctor')
@login_required
def doctor_dashboard():
    if current_user.role != 'doctor':
        return redirect(url_for('patient_dashboard'))

    # All patients
    all_patients = User.query.filter_by(role='patient').all()

    # Get latest prediction for each patient
    patient_data = []
    high_risk = moderate_risk = low_risk = 0

    for patient in all_patients:
        latest = Prediction.query.filter_by(user_id=patient.id)\
                 .order_by(Prediction.created_at.desc()).first()
        if latest:
            # Determine overall risk level
            max_risk = max(latest.heart_risk, latest.diabetes_risk, latest.kidney_risk)
            if max_risk >= 60:
                risk_cat = 'High Risk'; high_risk += 1
            elif max_risk >= 30:
                risk_cat = 'Moderate'; moderate_risk += 1
            else:
                risk_cat = 'Low Risk'; low_risk += 1
        else:
            risk_cat = 'No Data'

        patient_data.append({
            'patient': patient,
            'latest':  latest,
            'risk_cat': risk_cat,
        })

    # Sort — high risk first
    order = {'High Risk': 0, 'Moderate': 1, 'Low Risk': 2, 'No Data': 3}
    patient_data.sort(key=lambda x: order.get(x['risk_cat'], 4))

    # Recent predictions across all patients (last 10)
    recent_preds = Prediction.query.order_by(Prediction.created_at.desc()).limit(10).all()

    return render_template('doctor_dashboard.html',
        patient_data=patient_data,
        total=len(all_patients),
        high_risk=high_risk,
        moderate_risk=moderate_risk,
        low_risk=low_risk,
        recent_preds=recent_preds,
    )

@app.route('/doctor/patient/<int:patient_id>')
@login_required
def view_patient(patient_id):
    if current_user.role != 'doctor':
        return redirect(url_for('patient_dashboard'))

    patient = User.query.get_or_404(patient_id)
    predictions = Prediction.query.filter_by(user_id=patient_id)\
                  .order_by(Prediction.created_at.desc()).all()

    # Chart data
    chart_preds  = list(reversed(predictions[:6]))
    chart_labels = [p.created_at.strftime('%d %b') for p in chart_preds]
    chart_heart  = [p.heart_risk    for p in chart_preds]
    chart_diab   = [p.diabetes_risk for p in chart_preds]
    chart_kidney = [p.kidney_risk   for p in chart_preds]

    return render_template('view_patient.html',
        patient=patient,
        predictions=predictions,
        latest=predictions[0] if predictions else None,
        chart_labels=json.dumps(chart_labels),
        chart_heart=json.dumps(chart_heart),
        chart_diab=json.dumps(chart_diab),
        chart_kidney=json.dumps(chart_kidney),
    )

# ══════════════════════════════════════════════════
# API ROUTES
# ══════════════════════════════════════════════════

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data received'}), 400
        result = predict_all(data)
        return jsonify({'status': 'ok', 'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def api_health():
    metrics = json.load(open('models/metrics.json'))
    return jsonify({'status': 'ok', 'models': metrics})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  HealthGuard DB — Starting server...")
    print("  Open: http://localhost:5000")
    print("  Doctor login: doctor@healthguard.com / doctor123")
    print("="*50 + "\n")
    app.run(debug=False, host='0.0.0.0', port=5000)
