from flask import Flask, render_template, request
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import plotly.figure_factory as ff



app = Flask(__name__)

DATA_PATHS = {
    "Big 5 european leagues": r"C:\\Users\\LENOVO\\Downloads\\Dashboard Analyse du web\\static\\data\\Big5_Europian_leagues_final_dataset.csv",
    "Champions league": r"C:\\Users\\LENOVO\\Downloads\\Dashboard Analyse du web\\static\\data\\Champions_League_final_dataset.csv",
    "Europa league": r"C:\\Users\\LENOVO\\Downloads\\Dashboard Analyse du web\\static\\data\\Europa_league_final_dataset.csv",
    "Conference league": r"C:\\Users\\LENOVO\\Downloads\\Dashboard Analyse du web\\static\\data\\Conference_league_final_dataset.csv"}

def load_data(competition):
    path = DATA_PATHS.get(competition)
    if not path:
        return None
    return pd.read_csv(path)

def compute_kpis(df):
    return {
        "total_players": int(df.shape[0]),
        "avg_age": round(df["Age"].mean(), 1),
        "total_goals": int(df["Gls"].sum()),
        "total_assists": int(df["Ast"].sum())
    }

def generate_graphs(df, selected_player="Kylian Mbappé"):
    graphs = []

    if {"Player", "Min"}.issubset(df.columns):
        top_minutes = df[["Player", "Min"]].dropna()
        top_minutes = top_minutes.sort_values(by="Min", ascending=False).head(10)

        fig = px.bar(
            top_minutes,
            x="Player",
            y="Min",
            title="Top 10 joueurs par temps de jeu",
            color="Min",
            color_continuous_scale=px.colors.sequential.Greens
        )

        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            xaxis=dict(title="", showgrid=False),
            yaxis=dict(title="", showgrid=False),
            title=dict(x=0.5),
            showlegend=False
        )

        graphs.append(pio.to_html(fig, full_html=False, include_plotlyjs='cdn'))

    if {"Player", "Gls", "Ast"}.issubset(df.columns):
        df["G+A"] = df["Gls"] + df["Ast"]
        top10 = df.nlargest(10, "G+A")
        df_melt = top10.melt(id_vars="Player", value_vars=["Gls", "Ast"], var_name="Stat", value_name="Value")

        fig = px.bar(df_melt, x="Player", y="Value", color="Stat",
                     title="Top joueurs par buts et Assists",
                     color_discrete_map={"Gls": "#228b22", "Ast": "#00ff7f"})

        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", 
            paper_bgcolor="rgba(0,0,0,0)", 
            font=dict(color="white"),
            xaxis=dict(title="", showgrid=False),
            yaxis=dict(title="", showgrid=False),
            title=dict(x=0.5), legend=dict(title="")
        )

        graphs.append(pio.to_html(fig, full_html=False, include_plotlyjs='cdn'))

    if "Pos" in df.columns:
        position_counts = df["Pos"].value_counts().reset_index()
        position_counts.columns = ["Position", "Count"]

        fig = px.bar(position_counts, x="Count", y="Position", orientation="h", title="Distribution des positions",
                     color="Count", color_continuous_scale=px.colors.sequential.Greens)
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", 
            paper_bgcolor="rgba(0,0,0,0)", 
            font=dict(color="white", size=16),
            title_x=0.5,
            showlegend=False,
            xaxis_title='', yaxis_title='', xaxis_showticklabels=False
        )
        graphs.append(pio.to_html(fig, full_html=False, include_plotlyjs='cdn'))

    if {"Age", "Gls"}.issubset(df.columns):
        age_goals = df.groupby("Age")["Gls"].mean().reset_index()
        fig = px.line(age_goals, x="Age", y="Gls", title="Moyenne de buts par âge")
        fig.update_traces(
            line=dict(color="#00ff7f", width=3), 
            mode="lines+markers",
            marker=dict(color="white", size=6, line=dict(color="#00ff7f", width=2))
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", 
            paper_bgcolor="rgba(0,0,0,0)", 
            font=dict(color="white"),
            xaxis=dict(showgrid=False), 
            yaxis=dict(showgrid=False),
            title=dict(x=0.5),
            showlegend=False,
            xaxis_title='', yaxis_title='', yaxis_showticklabels=False
        )
        graphs.append(pio.to_html(fig, full_html=False, include_plotlyjs='cdn'))

    if {"PrgP", "Pos"}.issubset(df.columns):
        fig = px.box(df, x="Pos", y="PrgP", color="Pos", title="Passes progressives par position",
                     color_discrete_sequence=px.colors.sequential.Greens)
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", 
            paper_bgcolor="rgba(0,0,0,0)", 
            font=dict(color="white", size=14),
            title=dict(x=0.5), 
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            showlegend=False,
            xaxis_title='', yaxis_title='', yaxis_showticklabels=False
        )
        graphs.append(pio.to_html(fig, full_html=False, include_plotlyjs='cdn'))

    if "Nation" in df.columns:
        top_nations = df["Nation"].value_counts().nlargest(10).reset_index()
        top_nations.columns = ["Nationality", "Count"]
        fig = px.bar(top_nations, x="Nationality", y="Count", title="Les Nationalités les plus représentées",
                     color_discrete_sequence=["#228b22"])
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", 
            paper_bgcolor="rgba(0,0,0,0)", 
            font=dict(color="white"),
            xaxis=dict(showgrid=False), 
            yaxis=dict(showgrid=False),
            title=dict(x=0.5),
            showlegend=False,
            xaxis_title='', yaxis_title='', yaxis_showticklabels=False
        )
        graphs.append(pio.to_html(fig, full_html=False, include_plotlyjs='cdn'))

    if  {"Player", "Gls", "xG"}.issubset(df.columns):
        print("Création graphique Gls vs xG avec ligne y = x")
        fig = px.scatter(df, x="xG", y="Gls", text="Player",
                     labels={"xG": "xG", "Gls": "Buts Réels"},
                     title="Comparaison Buts Réels vs xG")
    
        fig.add_trace(go.Scatter(
            x=df["xG"], y=df["xG"], mode="lines", name="xG = Gls",
            line=dict(color="gray", dash="dash")
    ))

        fig.update_traces(marker=dict(size=10, color="lime"), textposition="top center")
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            title=dict(x=0.5),
            showlegend=False
    )
    graphs.append(pio.to_html(fig, full_html=False, include_plotlyjs='cdn'))

    # 🔍 Debug : Afficher le joueur sélectionné
    print("Joueur sélectionné pour le radar chart :", selected_player)

    # Heatmap de corrélation entre les stats (toujours affichée)
    stats = ["Gls", "Ast", "xG", "xAG", "PrgC", "PrgP", "PrgR"]
    if set(stats).issubset(df.columns):
        corr = df[stats].corr()
        fig_heatmap = ff.create_annotated_heatmap(
            z=corr.values.round(2),
            x=corr.columns.tolist(),
            y=corr.columns.tolist(),
            colorscale='Greens'
        )
        fig_heatmap.update_layout(
            title="Corrélation entre les statistiques",
            font=dict(color="white"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        graphs.append(pio.to_html(fig_heatmap, full_html=False, include_plotlyjs='cdn'))

    # Radar chart avec dropdown intégré pour choisir le joueur
    stats = ["Gls", "Ast", "xG", "xAG", "PrgC", "PrgP", "PrgR"]
    if set(stats).issubset(df.columns) and "Player" in df.columns:
        # Normalisation simple sur toutes les lignes pour ces colonnes
        df_norm = df.copy()
        for col in stats:
            max_val = df[col].max()
            df_norm[col] = df[col] / max_val if max_val != 0 else 0

        players = df["Player"].unique()

        fig = go.Figure()

        # Ajouter une trace par joueur (visible uniquement le 1er)
        for i, player in enumerate(players):
            player_vals = df_norm[df["Player"] == player][stats].iloc[0].values.tolist()
            fig.add_trace(go.Scatterpolar(
                r=player_vals,
                theta=stats,
                fill='toself',
                name=player,
                visible=(i == 0),
                line=dict(color="lime")
            ))

        # Dropdown menu pour sélectionner le joueur
        buttons = []
        for i, player in enumerate(players):
            visibility = [False] * len(players)
            visibility[i] = True
            buttons.append(dict(
                label=player,
                method="update",
                args=[{"visible": visibility}]
            ))
        

        fig.update_layout(
            updatemenus=[dict(
                active=0,
                buttons=buttons,
                x=0,
                y=1.15,
                xanchor='left',
                yanchor='top',
                bgcolor='rgba(0,0,0,0)',
                font=dict(color='Green'),
                bordercolor='gray',
                borderwidth=1
            )],
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            title="Profil de performance des joueurs",
        
        )

        graphs.append(pio.to_html(fig, full_html=False, include_plotlyjs='cdn'))

    return graphs

@app.route("/", methods=["GET", "POST"])
def index():
    competitions = list(DATA_PATHS.keys())
    selected = competitions[0]
    position_filter = None
    nation_filter = None
    team_filter = None

    if request.method == "POST":
        selected = request.form.get("competition", selected)
        position_filter = request.form.get("position")
        nation_filter = request.form.get("nation")
        team_filter = request.form.get("team")

    df = load_data(selected)

    if df is not None:
        if position_filter:
            df = df[df["Pos"] == position_filter]
        if nation_filter:
            df = df[df["Nation"] == nation_filter]
        if team_filter:
            df = df[df["Squad"] == team_filter]

        kpis = compute_kpis(df)
        graphs = generate_graphs(df)

        positions = sorted(df["Pos"].dropna().unique())
        nations = sorted(df["Nation"].dropna().unique())
        teams = sorted(df["Squad"].dropna().unique())
    else:
        kpis = {}
        graphs = []
        positions = []
        nations = []
        teams = []

    return render_template(
        "index.html",
        competitions=competitions,
        selected=selected,
        kpis=kpis,
        graphs=graphs,
        positions=positions,
        nations=nations,
        teams=teams,
        selected_position=position_filter,
        selected_nation=nation_filter,
        selected_team=team_filter
    )

if __name__ == "__main__":
    app.run(debug=True)