import logging


def apply_transformations(transform, results):
    """
    Führt definierte Transformationen auf den OCR-Ergebnissen aus.
    """
    # Extrahiere Werte in Dictionary (nur Strings!)
    values = {r["id"]: r["value"] for r in results if "id" in r and "value" in r}

    # Transformationen anwenden
    if transform == "balance_sum_zero":
        values_new = apply_balance_sum_zero(values)

        # Rückschreiben in results
        for r in results:
            if r["id"] in values_new:
                r["value"] = values_new[r["id"]]


def apply_balance_sum_zero(values):
    try:
        pv   = (lambda x: x / 1000 if x > 10 else x)(float(values["pv"]))
        bat  = (lambda x: x / 1000 if x > 10 else x)(float(values["bat"]))
        home = (lambda x: x / 1000 if x > 10 else x)(float(values["home"]))
        net  = (lambda x: x / 1000 if x > 10 else x)(float(values["net"]))
    except (ValueError, TypeError) as e:
        logging.warning(f"[Transform] Ungültiger Zahlenwert: {e}")
        return values

    # Beste Kombination der Vorzeichen für bat und net suchen, sodass: pv + bat + net ≈ home
    best_combo = None
    min_error = float("inf")

    for bat_sign in [1, -1]:
        for net_sign in [1, -1]:
            left = pv + bat * bat_sign + net * net_sign
            error = abs(left - home)
            if error < min_error:
                min_error = error
                best_combo = (bat_sign, net_sign)

    if best_combo:
        bat *= best_combo[0]
        net *= best_combo[1]
        logging.warning(f"[Transform] Bat={bat} Net={net}.")

    # Formatieren und -0.000 vermeiden
    def fmt(val):
        val = round(val, 3)
        return "0.000" if abs(val) < 0.001 else f"{val:.3f}"

    values["pv"] = fmt(pv)
    values["home"] = fmt(home)
    values["bat"] = fmt(bat)
    values["net"] = fmt(net)

    return values

