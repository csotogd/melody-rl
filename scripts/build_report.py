"""Build a portable static report with measured results, raw evidence and audio."""
import argparse
import html
import json
from pathlib import Path
import shutil


def copy_evidence(source, destination):
    if source.name == "summary.json":
        data = json.loads(source.read_text())
        project = Path(__file__).resolve().parent.parent
        for key in ("sft_adapter", "rl_adapter"):
            if data.get(key) and Path(data[key]).is_absolute():
                try:
                    data[key] = str(Path(data[key]).relative_to(project))
                except ValueError:
                    data[key] = Path(data[key]).name
        destination.write_text(json.dumps(data, indent=2) + "\n")
    else:
        shutil.copy2(source, destination)


def build(run, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    assets = output.parent / f"{output.stem}_assets"
    assets.mkdir(exist_ok=True)
    stages = [("base", "Modelo inicial"), ("sft-eval", "Después de SFT"),
              ("grpo-eval", "Después de SFT + GRPO"), ("rules", "Generador por reglas"),
              ]
    stages.extend((path.name, f"SFT · best-of-{path.name[6:]}") for path in sorted(run.glob("bestof*")) if path.is_dir())
    results = []
    for folder, label in stages:
        source = run / folder
        if not (source / "summary.json").exists():
            continue
        summary = json.loads((source / "summary.json").read_text())
        records = [json.loads(line) for line in (source / "predictions.jsonl").read_text().splitlines()]
        dest = assets / folder
        dest.mkdir(exist_ok=True)
        for name in ("summary.json", "predictions.jsonl"):
            copy_evidence(source / name, dest / name)
        if (source / "audio").exists():
            shutil.copytree(source / "audio", dest / "audio", dirs_exist_ok=True)
        results.append((folder, label, summary, records))
    if not results:
        raise ValueError("No completed evaluation stages found")
    # Refuse misleading side-by-side comparisons with different prompts.
    ids = [r["id"] for r in results[0][3]]
    if any([r["id"] for r in records] != ids for _, _, _, records in results):
        raise ValueError("Comparison requires identical task IDs in identical order")
    trainings = []
    for name in ("sft", "grpo"):
        directory = run / name
        if (directory / "summary.json").exists():
            summary = json.loads((directory / "summary.json").read_text())
            dest = assets / name
            dest.mkdir(exist_ok=True)
            for filename in ("summary.json", "train.jsonl", "rollouts.jsonl"):
                if (directory / filename).exists():
                    copy_evidence(directory / filename, dest / filename)
            trainings.append((name, summary))
    esc = html.escape
    lines = ['<!doctype html><html lang="es"><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width, initial-scale=1"><title>melody-rl · piloto local</title>',
             '<style>body{font:16px/1.6 system-ui,sans-serif;background:#f5f4ef;color:#182823;margin:0}main{max-width:1100px;margin:auto;padding:48px 24px}h1{font-size:48px;letter-spacing:-2px;margin:8px 0}h2{margin-top:40px}.tag{color:#496a5d;text-transform:uppercase;letter-spacing:2px;font-size:12px}.intro{max-width:750px}.notice{border-left:4px solid #b08330;padding:12px 18px;background:#fff9e9}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;background:white}th,td{text-align:left;padding:14px;border-bottom:1px solid #ddd}th{font-size:13px}code,pre{font:14px/1.5 ui-monospace,monospace}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef1ed;padding:12px;border-radius:8px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:16px}.card{background:white;padding:20px;border:1px solid #ddd;border-radius:12px}.pass{color:#276848}.fail{color:#9c492e}audio{width:100%;height:36px}a{color:#265a4a}.muted{color:#59685f;font-size:14px}details{margin:12px 0}footer{margin-top:48px;border-top:1px solid #ccc;padding-top:16px}</style><main>',
             '<div class="tag">Experimento 001 · entrenamiento local</div><h1>melody-rl</h1>',
             '<p class="intro">¿Puede el post-training mejorar el cumplimiento de instrucciones musicales? Qwen3-0.6B + LoRA + GRPO, con recompensas verificables y melodías cortas.</p>',
             '<p class="notice"><strong>Piloto de viabilidad.</strong> Esta muestra de validación es demasiado pequeña para concluir que RL mejora el modelo. El cumplimiento de restricciones no mide calidad musical. El audio es una síntesis sencilla de las notas, no audio generado por el modelo.</p>',
             '<p class="intro"><strong>SFT (ajuste fino supervisado)</strong>: aprende de ejemplos de melodías correctas. <strong>GRPO</strong>: es el método de aprendizaje por refuerzo (RL) que usamos después; compara melodías propuestas por el modelo y actualiza sus pesos según las reglas que cumplen. <strong>Best-of-N</strong>: genera N alternativas y elige la mejor puntuación, sin entrenamiento adicional.</p>',
             '<h2>Resultados medidos</h2><div class="scroll"><table><thead><tr><th>Método</th><th>Todas las restricciones</th><th>Formato válido</th><th>Restricciones cumplidas, media</th><th>Repetición consecutiva¹</th></tr></thead><tbody>']
    for folder, label, s, _ in results:
        repeat = s["repeat_rate_valid_outputs"]
        low, high = s["wilson_95_interval"]
        lines.append(f'<tr><td>{esc(label)}</td><td>{s["successes"]}/{s["n"]} <span class="muted">(IC 95%: {low:.0%}–{high:.0%})</span></td><td>{s["valid_rate"]:.0%}</td><td>{s["mean_partial_all_outputs"]:.0%}</td><td>{repeat:.0%}</td></tr>' if repeat is not None else f'<tr><td>{esc(label)}</td><td>{s["successes"]}/{s["n"]}</td><td>{s["valid_rate"]:.0%}</td><td>{s["mean_partial_all_outputs"]:.0%}</td><td>Sin salidas válidas</td></tr>')
    lines.append('</tbody></table></div><p class="muted">¹ Solo salidas válidas. El generador por reglas garantiza las restricciones por construcción: es un control, no una salida del modelo.</p>')
    lines.append('<details><summary>Coste de generación de esta evaluación</summary><div class="scroll"><table><thead><tr><th>Método</th><th>Tokens generados</th><th>Tiempo de generación</th><th>Tiempo total</th></tr></thead><tbody>')
    for _, label, summary, _ in results:
        lines.append(f'<tr><td>{esc(label)}</td><td>{summary["generated_tokens"]:,}</td><td>{summary["generation_seconds"]:.1f} s</td><td>{summary["seconds"]:.1f} s</td></tr>')
    lines.append('</tbody></table></div><p class="muted">Best-of-N cuenta las N alternativas, incluidas las descartadas. El tiempo total incluye carga del modelo y exportación de audio. El generador por reglas no usa tokens del modelo; su tiempo de generación no se instrumenta por separado.</p></details>')
    if trainings:
        lines.append('<h2>Entrenamiento ejecutado</h2><div class="grid">')
        for name, summary in trainings:
            lines.append(f'<div class="card"><strong>{name.upper()}</strong><p>{summary["updates"]} actualizaciones · {summary["seconds"]:.1f} s</p><p class="muted">{summary["trainable_parameters"]:,} parámetros entrenables<br>RSS pico del proceso: {summary["process_peak_rss_gb"]:.2f} GiB<br>Dispositivo: {esc(summary["device"])} · nube: 0 €</p></div>')
        lines.append('</div><p class="muted">Los tiempos incluyen carga del modelo y excluyen la descarga inicial. RSS no representa toda la memoria unificada de la GPU.</p>')
    lines.append('<h2>Escucha e inspecciona</h2>')
    for index, first in enumerate(results[0][3]):
        lines.append(f'<details {"open" if index == 0 else ""}><summary>Ejemplo {index + 1} · {esc(first["id"])}</summary><p class="muted">{esc(first["prompt"])}</p><div class="grid">')
        for folder, label, _, records in results:
            record = records[index]
            metrics = record["metrics"]
            cls = "pass" if metrics["all_satisfied"] else "fail"
            verdict = "Cumple todo" if metrics["all_satisfied"] else "Incumple restricciones"
            score_text = esc(record["text"]).replace(" \n", "&#32;\n")
            lines.append(f'<article class="card"><strong>{esc(label)}</strong><p class="{cls}">{verdict}</p><pre>{score_text}</pre>')
            failed = [key for key, value in metrics["checks"].items() if not value]
            if failed:
                lines.append(f'<p class="muted">Fallos: {esc(", ".join(failed))}</p>')
            if metrics["valid"]:
                prefix = f'{assets.name}/{folder}/audio/{index:03d}'
                lines.append(f'<audio controls preload="none" src="{esc(prefix)}.wav"></audio><a href="{esc(prefix)}.mid" download>MIDI</a>')
            lines.append('</article>')
        lines.append('</div></details>')
    lines.append('<footer><strong>Evidencia reproducible</strong><p>Los registros contienen todas las salidas, configuraciones y métricas, incluidos los fallos.</p><ul>')
    for folder, label, _, _ in results:
        lines.append(f'<li>{esc(label)}: <a href="{assets.name}/{folder}/summary.json">métricas</a> · <a href="{assets.name}/{folder}/predictions.jsonl">predicciones</a></li>')
    for name, _ in trainings:
        lines.append(f'<li>{name.upper()}: <a href="{assets.name}/{name}/train.jsonl">registro de entrenamiento</a></li>')
    lines.append('</ul><p class="muted">Datos sintéticos propios · modelo Qwen3-0.6B · código MIT · no se han evaluado los conjuntos de test reservados.</p></footer></main></html>')
    output.write_text('\n'.join(lines))
    print(output.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.run, args.output)
