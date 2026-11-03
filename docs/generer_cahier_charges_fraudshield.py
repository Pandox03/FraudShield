"""Génère le cahier des charges PDF du Projet 3 — FraudShield."""

from pathlib import Path

from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).resolve().parent / "CAHIER_DES_CHARGES_PROJET3_FraudShield.pdf"


def styles():
    base = getSampleStyleSheet()
    return {
        "left_h": ParagraphStyle(
            "left_h",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=11,
            leading=14,
            alignment=TA_LEFT,
        ),
        "right_h": ParagraphStyle(
            "right_h",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=11,
            leading=14,
            alignment=TA_RIGHT,
        ),
        "title": ParagraphStyle(
            "title",
            parent=base["Normal"],
            fontName="Times-Bold",
            fontSize=14,
            leading=18,
            alignment=TA_CENTER,
            spaceBefore=28,
            spaceAfter=18,
        ),
        "h": ParagraphStyle(
            "h",
            parent=base["Normal"],
            fontName="Times-Bold",
            fontSize=12,
            leading=16,
            spaceBefore=14,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=11,
            leading=15,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
        ),
        "item": ParagraphStyle(
            "item",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=11,
            leading=15,
            leftIndent=18,
            spaceAfter=3,
        ),
    }


def main():
    s = styles()
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=2.2 * cm,
        rightMargin=2.2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="Cahier des charges — Projet 3 FraudShield",
    )

    story = []

    story.append(
        Paragraph(
            "Module : Développement et déploiement d'applications intelligentes",
            s["left_h"],
        )
    )
    story.append(Paragraph("Pr. Soukaina Gouraguine", s["right_h"]))
    story.append(Spacer(1, 20))

    story.append(Paragraph("CAHIER DES CHARGES DES PROJETS DE GROUPE", s["title"]))

    story.append(
        Paragraph(
            "Réalisé par : Imad Guidouh, Zakaria Bejgana, Zini Ayoub,<br/>Fathallah Zyad, Zakraia Tattah",
            ParagraphStyle(
                "auteurs",
                parent=s["body"],
                alignment=TA_CENTER,
                spaceBefore=0,
                spaceAfter=16,
            ),
        )
    )

    story.append(
        Paragraph(
            "PROJET 3 : FraudShield — Détection des transactions bancaires frauduleuses",
            s["h"],
        )
    )

    story.append(Paragraph("1. Contexte et objectif", s["h"]))
    story.append(
        Paragraph(
            "Le projet FraudShield consiste à développer une application intelligente qui "
            "exploite des données de transactions bancaires afin d'identifier des opérations "
            "présentant des signes de fraude et de fournir des indicateurs utiles à leur analyse.",
            s["body"],
        )
    )
    story.append(
        Paragraph(
            "L'application doit intégrer un modèle de machine learning, une API et une "
            "interface utilisateur. Elle constitue un outil d'aide à l'analyse, et non un "
            "système de décision automatique concernant les transactions.",
            s["body"],
        )
    )

    story.append(Paragraph("2. Données et technologies", s["h"]))
    story.append(
        Paragraph(
            "• Jeu de données : Credit Card Fraud Detection (Kaggle / ULB).",
            s["item"],
        )
    )
    story.append(Paragraph("• Langage : Python.", s["item"]))
    story.append(
        Paragraph(
            "• Traitement et machine learning : Pandas, Scikit-learn.",
            s["item"],
        )
    )
    story.append(Paragraph("• API : Flask ou FastAPI.", s["item"]))
    story.append(
        Paragraph(
            "• Interface : Streamlit ou une solution équivalente.",
            s["item"],
        )
    )
    story.append(
        Paragraph(
            "• Tests et déploiement : outils adaptés aux notions étudiées dans le module.",
            s["item"],
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "Les technologies proposées peuvent être remplacées par des solutions "
            "équivalentes, à condition de justifier les choix.",
            s["body"],
        )
    )

    story.append(Paragraph("3. Fonctionnalités minimales attendues", s["h"]))
    story.append(
        Paragraph("1. Charger, nettoyer et analyser les données.", s["item"])
    )
    story.append(
        Paragraph(
            "2. Entraîner et comparer au moins deux modèles de classification.",
            s["item"],
        )
    )
    story.append(
        Paragraph(
            "3. Évaluer le modèle retenu à l'aide de métriques adaptées "
            "(notamment face au déséquilibre des classes).",
            s["item"],
        )
    )
    story.append(
        Paragraph("4. Exposer le modèle à travers une API.", s["item"])
    )
    story.append(
        Paragraph(
            "5. Développer une interface permettant de consulter les indicateurs "
            "et de tester une prédiction.",
            s["item"],
        )
    )
    story.append(
        Paragraph(
            "6. Tester et documenter l'application afin de permettre sa reproduction.",
            s["item"],
        )
    )

    story.append(Paragraph("4. Travail pédagogique demandé", s["h"]))
    story.append(Paragraph("Le groupe doit également préparer :", s["body"]))
    story.append(
        Paragraph(
            "• Un support de cours présentant les concepts et les technologies mobilisés.",
            s["item"],
        )
    )
    story.append(Paragraph("• Un TP guidé avec son corrigé.", s["item"]))
    story.append(
        Paragraph(
            "• Une démonstration du prototype et une explication des choix techniques.",
            s["item"],
        )
    )

    story.append(Spacer(1, 18))
    story.append(Paragraph("Livrables attendus", s["h"]))
    story.append(Paragraph("Chaque groupe doit remettre :", s["body"]))
    story.append(
        Paragraph(
            "1. Support pédagogique : présentation expliquant les concepts, "
            "l'architecture, les technologies, les exemples et les limites.",
            s["item"],
        )
    )
    story.append(
        Paragraph(
            "2. Code source : fichiers nécessaires à l'exécution du prototype.",
            s["item"],
        )
    )
    story.append(
        Paragraph(
            "3. TP : objectifs, prérequis, étapes, manipulations et résultats attendus.",
            s["item"],
        )
    )
    story.append(
        Paragraph(
            "4. Correction du TP : solution exécutable et explication des résultats.",
            s["item"],
        )
    )

    doc.build(story)
    print(f"PDF généré : {OUT}")


if __name__ == "__main__":
    main()
