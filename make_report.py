from pathlib import Path
import json
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak


def build_report(root):
    root = Path(root)
    results = root / 'results'
    summary = json.loads((results / 'summary.json').read_text())
    evaluation = pd.read_csv(results / 'final_metrics.csv')
    errors = pd.read_csv(results / 'test_predictions.csv').nlargest(3, 'absolute_error')
    styles = getSampleStyleSheet()
    styles['BodyText'].fontSize = 9.4
    styles['BodyText'].leading = 12.5
    story = []

    def paragraph(text, style='BodyText'):
        story.extend([Paragraph(text, styles[style]), Spacer(1, 6)])

    def table(rows, widths):
        item = Table(rows, colWidths=widths, hAlign='LEFT')
        item.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dcebf0')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f3f6f7')]),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5)]))
        story.extend([item, Spacer(1, 12)])

    def image(name, width=495):
        from PIL import Image as PILImage
        path = results / 'figures' / name
        with PILImage.open(path) as pixels:
            height = width * pixels.height / pixels.width
        story.extend([Image(str(path), width=width, height=height), Spacer(1, 10)])

    paragraph('Car price prediction', 'Title')
    paragraph('Research AI Task 6 | Gradient descent from scratch', 'Heading2')
    paragraph('The notebook now trains one linear regression model, with a log-price target and ridge penalty 10. '
              'These settings were selected by the previous training-only search. Removing the general search framework '
              'retains the model and its performance.')
    paragraph('What the project does', 'Heading2')
    paragraph('Manufacturer names are corrected, and the row ID and full car name are excluded. '
              'The 205 cars are split into 164 training rows and 41 test rows, with identical predictor groups kept together. '
              'Encoding, imputation and scaling use training rows only. Five grouped validation folds are repeated three times.')
    paragraph('All coefficients are learned through NumPy gradient updates. Momentum speeds convergence; the intercept is '
              'not regularized. The model converged in 225 updates. A training-residual correction converts log predictions '
              'back to price units. No scikit-learn or direct regression solver is used.')
    paragraph('Evaluation metrics', 'Heading2')
    rows = [['Metric', 'Training', 'Cross-validation', 'Test']]
    for key, label in [('MAE', 'MAE'), ('MSE', 'MSE'), ('RMSE', 'RMSE'), ('R2', 'R-squared'),
                       ('MAPE_pct', 'MAPE (%)'), ('sMAPE_pct', 'sMAPE (%)'), ('MedianAE', 'Median absolute error')]:
        fmt = '.4f' if key == 'R2' else ',.2f'
        rows.append([label] + [format(summary[part][key], fmt) for part in ['train_metrics', 'cv_metrics', 'test_metrics']])
    table(rows, [170, 108, 108, 108])
    paragraph('MAE, RMSE and median error use price units; MSE uses squared units. Percentage errors use percent. '
              'The supplied dictionary does not identify a currency. Lower errors and higher R-squared are better.')
    image('test_diagnostics.png')

    story.append(PageBreak())
    paragraph('Errors and verification', 'Title')
    rows = [['Test model', 'RMSE', 'MAE']]
    for row in evaluation[evaluation.Split == 'Test'].itertuples():
        rows.append([row.Model, f'{row.RMSE:,.2f}', f'{row.MAE:,.2f}'])
    table(rows, [280, 107, 107])
    paragraph('The selected model reduces test RMSE by 76.1% relative to predicting the training mean. '
              'Positive residuals mean underprediction. All cars, including expensive cars, remain in the evaluation.')
    image('training_loss.png', 380)
    paragraph('Largest test errors', 'Heading2')
    rows = [['Car', 'Actual', 'Predicted', 'Absolute error']]
    for row in errors.itertuples():
        rows.append([row.CarName, f'{row.actual:,.0f}', f'{row.predicted:,.0f}', f'{row.absolute_error:,.0f}'])
    table(rows, [225, 88, 88, 93])
    paragraph('Verification', 'Heading2')
    paragraph('The simplified model reproduces the previous predictions within numerical roundoff, with the same reported '
              'metrics. Tests cover a known synthetic line, ridge shrinkage, manually calculated metrics, missing and unseen '
              'categories, and disjoint data splits. The notebook runs from a fresh kernel.')
    paragraph('This small historical dataset cannot establish accuracy for current market prices or unseen manufacturers. '
              'Cross-validation variability and the train/test error gap describe uncertainty in these results.')
    paragraph('<link href="https://github.com/sooryas1233213/task-phase">Project repository</link>')

    def footer(canvas, document):
        canvas.setFont('Helvetica', 8)
        canvas.drawString(45, 25, 'Research AI Task 6 | Car price regression')
        canvas.drawRightString(A4[0]-45, 25, str(document.page))

    output = root / 'output' / 'pdf' / 'car_price_report.pdf'
    output.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(output), pagesize=A4, leftMargin=45, rightMargin=45,
                      topMargin=35, bottomMargin=40, title='Car price prediction').build(
                          story, onFirstPage=footer, onLaterPages=footer)
    return output


if __name__ == '__main__':
    print(build_report(Path(__file__).resolve().parent))
