"""Build the public experiment overview from exported measurements only."""
import argparse
import html
import json
from pathlib import Path


def build(directory):
    measured = json.loads((directory / 'study-summary.json').read_text())
    path = directory / 'generalization-summary.json'
    generalization = json.loads(path.read_text()) if path.exists() else {'splits': {}}
    splits = {
        'test_id': ('Tareas nuevas, familias conocidas', measured['splits']['test_id'], 'extended-test.html',
                    '48 combinaciones de restricciones que no aparecieron en el entrenamiento.'),
        'test_composition': ('Combinación no vista', generalization['splits'].get('test_composition'), 'generalization-composition.html',
                            '54 tareas en Sol mayor y compás 4/4. El modelo vio ambos componentes, pero no esta combinación.'),
        'test_length': ('Melodías más largas', generalization['splits'].get('test_length'), 'generalization-length.html',
                       '162 tareas de tres compases. El entrenamiento solo incluía uno o dos; este conjunto también incluye la combinación Sol mayor + 4/4.'),
    }
    labels = {'sft': 'SFT', 'grpo': 'SFT + RL · reglas', 'graded': 'SFT + RL · reglas', 'diverse': 'SFT + RL · diversidad'}
    esc = html.escape
    parts = ['''<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>melody-rl · qué aprende y dónde falla</title><style>
body{font:16px/1.6 system-ui,sans-serif;color:#182823;background:#f5f4ef;margin:0}main{max-width:1100px;margin:auto;padding:40px 24px}h1{font-size:42px;line-height:1.15;letter-spacing:-1.4px}h2{margin-top:44px}.tag{font-size:12px;text-transform:uppercase;letter-spacing:2px;color:#496a5d}.note{padding:16px 20px;border-left:4px solid #b08330;background:#fff9e9}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;background:white}th,td{padding:13px;border-bottom:1px solid #ddd;text-align:left}th{font-size:13px}a{color:#265a4a}.muted{color:#59685f;font-size:14px}.links{display:flex;flex-wrap:wrap;gap:16px}details{margin-top:18px}footer{margin-top:48px;border-top:1px solid #ccc;padding-top:16px}code{font-size:14px}</style>
<main><div class="tag">Experimento ejecutado · Apple M4 · 0 € de nube</div>
<h1>Qué aprende RL<br>y dónde deja de funcionar</h1>
<p>Qwen3-0.6B aprende a escribir melodías simbólicas que deben cumplir seis reglas. Primero aprende de ejemplos correctos (SFT); después comparamos dos recompensas de aprendizaje por refuerzo (GRPO).</p>
<p class="note"><strong>Mejor cumplimiento no significa mejor música.</strong> La recompensa original favorece soluciones repetitivas. Añadir una señal de diversidad reduce ese problema, pero no elimina las plantillas. El audio es una síntesis sencilla de notas.</p>
<div class="links"><a href="https://github.com/csotogd/melody-rl">Repositorio público</a><a href="extended-validation.html">Validación y audios</a><a href="diagnosis/generalization-protocol.json">Protocolo de generalización</a></div>
<p class="muted">Métrica principal: todas las muestras generadas, antes de seleccionar. Cuatro alternativas por tarea, temperatura 1. Los tres modelos estaban fijados antes de abrir los tests; no se reentrenaron al ver sus resultados.</p>''']
    for split, (title, rows, report, description) in splits.items():
        parts.append(f'<section id="{split}"><h2>{esc(title)}</h2><p>{esc(description)}</p>')
        if rows is None:
            parts.append('<p>Evaluación en curso; aún no hay resultados completos.</p></section>')
            continue
        parts.append('<div class="scroll"><table><thead><tr><th>Modelo</th><th>Muestras correctas</th><th>Repetición¹</th><th>Generación directa</th><th>Elegir entre cuatro</th></tr></thead><tbody>')
        for policy, row in rows.items():
            if policy == 'rules':
                continue
            repetition = row['sampled_repetition_mean_per_task']
            repeat_label = f'{repetition:.1%}' if repetition is not None else 'Sin salidas válidas'
            parts.append(f'<tr><td>{esc(labels[policy])}</td><td><strong>{row["sampled_success_rate"]:.1%}</strong> ({row["sampled_successes"]}/{row["sampled_completions"]})</td><td>{repeat_label}</td><td>{row["greedy_successes"]}/{row["tasks"]}</td><td>{row["best_of_four_successes"]}/{row["tasks"]}</td></tr>')
        parts.append(f'</tbody></table></div><p><a href="{report}">Escuchar y comparar todas las tareas →</a></p>')
        if split == 'test_length':
            row = rows['diverse']
            parts.append(f'<p class="note">Con diversidad, la generación directa resuelve <strong>{row["greedy_successes"]}/{row["tasks"]}</strong> tareas largas, pero solo el <strong>{row["sampled_success_rate"]:.1%}</strong> de las muestras aleatorias cumple todo. La transferencia es parcial y depende del modo de generar; elegir entre cuatro resuelve {row["best_of_four_successes"]}/{row["tasks"]}.</p>')
        if 'rules' in rows:
            rule = rows['rules']
            parts.append(f'<p class="muted">Control por reglas: {rule["successes"]}/{rule["tasks"]}. Resuelve la tarea por construcción; no es una salida del modelo.</p>')
        if split != 'test_id':
            parts.append('<details><summary>Qué reglas incumplen las muestras</summary><p class="muted">Porcentaje de muestras que falla cada regla. Un mismo error puede afectar varias columnas; las salidas inválidas fallan todas.</p><div class="scroll"><table><thead><tr><th>Modelo</th><th>Compases</th><th>Notas por compás</th><th>Tonalidad</th><th>Rango</th><th>Final</th><th>Saltos</th></tr></thead><tbody>')
            for policy, row in rows.items():
                if policy == 'rules':
                    continue
                failures = ''.join(f'<td>{1-row["sampled_constraint_pass_rates"][k]:.1%}</td>' for k in ('bars','meter','key','range','final','leap'))
                parts.append(f'<tr><td>{esc(labels[policy])}</td>{failures}</tr>')
            parts.append('</tbody></table></div></details>')
        parts.append('</section>')
    parts.append('''<p class="muted">¹ Media por tarea de la proporción de notas consecutivas iguales en las salidas válidas. Una melodía alternante puede repetir una plantilla sin aumentar esta métrica. Best-of-4 usa el mismo selector por reglas para todos los modelos.</p>
<h2>Qué permite afirmar este experimento</h2>
<p>El proyecto muestra entrenamiento real, evaluación separada, un fallo de la recompensa y una comparación controlada para mitigarlo. Los resultados de los distintos conjuntos se mantienen separados: una mejora en tareas conocidas no garantiza extrapolación a más compases.</p>
<p>Solo se ha entrenado con una semilla. Los intervalos por tarea están en los informes y no miden variación entre entrenamientos. No se han ejecutado controles de más SFT con el mismo cómputo, otras temperaturas ni entrenamiento distribuido. Las reglas de tonalidad permiten usar notas comunes a todas las escalas.</p>
<footer><strong>Evidencia y reproducción</strong><p><a href="study-summary.json">Resultados ID y validación</a> · <a href="generalization-summary.json">Resultados de generalización</a> · <a href="generalization-experiment.json">Configuración, comandos y huellas de los modelos</a> · <a href="generalization-diagnostics.json">Diagnóstico de errores</a> · <a href="README.md">Notas completas</a></p><p class="muted">Los tests ejecutados quedan consumidos. Cualquier desarrollo posterior debe declarar ese uso y reservar nuevos datos para una nueva evaluación final.</p></footer></main></html>''')
    output = directory / 'index.html'
    output.write_text('\n'.join(parts))
    print(output.resolve())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=Path('results'))
    build(parser.parse_args().results)
