import streamlit as st
import pandas as pd
import numpy as np
import random
import uuid
import plotly.graph_objects as go
from datetime import datetime

# Setup page config
st.set_page_config(
    page_title="Guitar Hero Tournament Manager",
    page_icon="🎸",
    layout="wide",
)

# Initialize session state variables if they don't exist
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.players = []
    st.session_state.beginner_bracket = []
    st.session_state.casual_bracket = []
    st.session_state.competitive_bracket = []
    st.session_state.beginner_matches = []
    st.session_state.casual_matches = []
    st.session_state.competitive_matches = []
    st.session_state.tournament_started = False
    st.session_state.tournament_log = []
    st.session_state.beginner_rounds = 0
    st.session_state.casual_rounds = 0
    st.session_state.competitive_rounds = 0
    st.session_state.tournament_format = "standard"  # or "skill_based"
    # Add state to track last BYE recipient for fairness in skill-based mode
    st.session_state.beginner_last_bye = None
    st.session_state.casual_last_bye = None
    st.session_state.competitive_last_bye = None


# --- Helper Functions ---

def log_event(message):
    """Adds a timestamped message to the tournament log."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    # Ensure log entries are strings
    st.session_state.tournament_log.insert(0, f"[{timestamp}] {str(message)}")

def generate_match_id():
    """Generates a unique short ID for a match."""
    return str(uuid.uuid4())[:8]

def create_match(player1, player2, bracket_type, round_num=1):
    """Creates a dictionary representing a match."""
    match_id = generate_match_id()
    match = {
        'match_id': match_id,
        'player1': player1,
        'player2': player2,
        'winner': None,
        'player1_score': 0,
        'player2_score': 0,
        'status': 'Pending',  # 'Pending', 'Completed'
        'bracket': bracket_type, # 'Beginner', 'Casual', 'Competitive'
        'round': round_num,
        'created_at': datetime.now().strftime("%H:%M:%S")
    }
    return match

def check_round_completion(matches, round_num):
    """Checks if all matches in a specific round are completed."""
    round_matches = [m for m in matches if m['round'] == round_num]
    completed_round_matches = [m for m in round_matches if m['status'] == 'Completed']
    # A round is complete if all its matches are completed
    return round_matches and len(round_matches) == len(completed_round_matches)

def update_rounds_counter(bracket_type, round_num):
    """Updates the maximum round number reached for a bracket."""
    if bracket_type == "Beginner":
        st.session_state.beginner_rounds = max(st.session_state.beginner_rounds, round_num)
    elif bracket_type == "Casual":
        st.session_state.casual_rounds = max(st.session_state.casual_rounds, round_num)
    else: # Competitive
        st.session_state.competitive_rounds = max(st.session_state.competitive_rounds, round_num)

def get_players_by_status(bracket_type, status, exclude_pending=True):
    """Retrieves players from a bracket based on their status."""
    # Note: This function seems unused in the provided code flow but kept for potential future use.
    if bracket_type == "Beginner":
        bracket = st.session_state.beginner_bracket
        matches = st.session_state.beginner_matches
    elif bracket_type == "Casual":
        bracket = st.session_state.casual_bracket
        matches = st.session_state.casual_matches
    else: # Competitive
        bracket = st.session_state.competitive_bracket
        matches = st.session_state.competitive_matches

    if exclude_pending:
        return [p for p in bracket if p['status'] == status and
                not any(m['status'] == 'Pending' and
                        (m['player1'] == p['name'] or m['player2'] == p['name']) for m in matches)]
    else:
        return [p for p in bracket if p['status'] == status]

def get_player_skill(player_name):
    """Calculates a player's skill based on win rate (0-100). Returns 50 if no games played or player not found."""
    if player_name == "BYE": return -1 # Should not happen in skill calculation context

    for player in st.session_state.players:
        if player['name'] == player_name:
            wins = int(player.get('wins', 0))
            losses = int(player.get('losses', 0))
            total_games = wins + losses
            if total_games == 0:
                return 50 # Neutral skill for players with no match history
            return (wins / total_games) * 100
    return 50 # Default if player somehow not found

def update_player_stats(player_name, score, is_winner, bracket_type):
    """Updates a player's global stats (wins, losses, scores) and their status within a specific bracket."""
    if player_name == "BYE": return # Do not update stats for BYE

    # Update global player stats
    player_updated = False
    for player in st.session_state.players:
        if player['name'] == player_name:
            player['total_score'] = int(player.get('total_score', 0)) + int(score)
            player['wins'] = int(player.get('wins', 0))
            player['losses'] = int(player.get('losses', 0))

            if is_winner:
                player['wins'] += 1
            else:
                player['losses'] += 1

            total_matches = player['wins'] + player['losses']
            player['avg_score'] = player['total_score'] / total_matches if total_matches > 0 else 0
            player_updated = True
            break

    if not player_updated:
         log_event(f"Warning: Could not find player {player_name} in global list to update stats.")


    # Update player status within the specific bracket list (more relevant for progression)
    bracket_list = None
    if bracket_type == "Beginner": bracket_list = st.session_state.beginner_bracket
    elif bracket_type == "Casual": bracket_list = st.session_state.casual_bracket
    else: bracket_list = st.session_state.competitive_bracket

    if bracket_list is not None:
        bracket_player_updated = False
        for p_in_bracket in bracket_list:
            if p_in_bracket['name'] == player_name:
                # Status reflects outcome of the last match in this bracket
                p_in_bracket['status'] = 'Winner' if is_winner else 'Loser'
                bracket_player_updated = True
                break


# --- Tournament Logic Functions ---

# MODIFIED function with random BYE assignment and skill-based pairing after BYE
def create_skill_based_matches(bracket_type):
    """
    Creates matches for the next round in skill-based format.
    Assigns a BYE randomly to an eligible player if the count is odd.
    Pairs remaining players based on closest skill after BYE assignment.
    """
    # Determine the correct bracket, matches list, and round info
    last_bye_state_key = f"{bracket_type.lower()}_last_bye"
    if bracket_type == "Beginner":
        bracket = st.session_state.beginner_bracket
        matches = st.session_state.beginner_matches
        current_round = st.session_state.beginner_rounds + 1
        last_bye_recipient = st.session_state.beginner_last_bye
    elif bracket_type == "Casual":
        bracket = st.session_state.casual_bracket
        matches = st.session_state.casual_matches
        current_round = st.session_state.casual_rounds + 1
        last_bye_recipient = st.session_state.casual_last_bye
    else: # Competitive
        bracket = st.session_state.competitive_bracket
        matches = st.session_state.competitive_matches
        current_round = st.session_state.competitive_rounds + 1
        last_bye_recipient = st.session_state.competitive_last_bye

    # Find available players (in bracket, not in pending matches)
    player_names_in_bracket = {p['name'] for p in bracket}
    pending_player_names = {m[p] for m in matches if m['status'] == 'Pending' for p in ['player1', 'player2'] if m[p] != "BYE" and m[p] in player_names_in_bracket}
    available_players = [p['name'] for p in bracket if p['name'] not in pending_player_names]

    if not available_players:
        log_event(f"No available players for new {bracket_type} matches in Round {current_round}.")
        return False

    # --- BYE Assignment (if odd number of players) ---
    match_created_flag = False
    bye_player = None
    players_to_pair = list(available_players) # Copy list for pairing

    if len(players_to_pair) % 2 != 0:
        # Try to find players who did NOT get the last bye
        eligible_for_bye = [p for p in players_to_pair if p != last_bye_recipient]

        if eligible_for_bye:
            # Assign BYE randomly from eligible players
            bye_player = random.choice(eligible_for_bye)
            log_event(f"Randomly selected {bye_player} for BYE from eligible players in {bracket_type} Round {current_round}.")
        elif players_to_pair: # Only one player available, or all available got the last bye
            # Assign BYE randomly from all available players (unavoidable repeat possible)
            bye_player = random.choice(players_to_pair)
            log_event(f"Note: unavoidable random BYE assignment to {bye_player} in {bracket_type} Round {current_round} (all available had last BYE or only one player).")
        else: # Should not happen if available_players was not empty initially
             log_event(f"Error: Could not determine BYE player for {bracket_type} Round {current_round}.")
             return False # Exit if error state

        # Remove the selected BYE player from the list to be paired
        players_to_pair.remove(bye_player)

        # Create the BYE match record
        bye_match = create_match(bye_player, "BYE", bracket_type, current_round)
        bye_match['status'] = 'Completed'
        bye_match['winner'] = bye_player
        bye_match['player1_score'] = 1
        bye_match['player2_score'] = 0
        matches.append(bye_match)
        # log_event(f"{bracket_type} player {bye_player} gets a BYE in Round {current_round}") # Logged above based on selection method

        # Update stats and track the BYE recipient
        update_player_stats(bye_player, 1, True, bracket_type)
        setattr(st.session_state, last_bye_state_key, bye_player) # Update last bye state
        match_created_flag = True

    # --- Pair Remaining Players ---
    if len(players_to_pair) >= 2:
        # Sort the remaining players by skill for pairing
        players_to_pair.sort(key=get_player_skill, reverse=True)

        # Pair sequentially (highest remaining vs next highest, etc.)
        for i in range(0, len(players_to_pair), 2):
             # Ensure there's a pair
             if i + 1 < len(players_to_pair):
                 player1 = players_to_pair[i]
                 player2 = players_to_pair[i+1]
                 new_match = create_match(player1, player2, bracket_type, current_round)
                 matches.append(new_match)
                 log_event(f"Created skill-based {bracket_type} match: {player1} vs {player2} (Round {current_round})")
                 match_created_flag = True
             else:
                 # This should not happen if we start with an even number after BYE removal
                 log_event(f"Warning: Odd player {players_to_pair[i]} left over after BYE removal in {bracket_type} Round {current_round}. Waiting.")


    # Update round counter if any match/BYE was created
    if match_created_flag:
        update_rounds_counter(bracket_type, current_round)
        return True
    elif not bye_player and not players_to_pair: # No players were available initially
        # Log message already handled at the start
        return False
    else: # Only a BYE player was assigned, no pairs made (shouldn't happen if >= 1 player) or only 1 player remained after BYE
        # Check if players_to_pair is not empty before accessing index 0
        player_left_over = players_to_pair[0] if players_to_pair else 'N/A'
        log_event(f"Only one player ({player_left_over}) available after BYE assignment in {bracket_type} Round {current_round}. Waiting for more players.")
        # If only a BYE was created, the round counter was already updated
        return match_created_flag # Return True if BYE was made, False otherwise


def create_next_round_matches(match):
    """
    Creates matches for the next round in a standard knockout tournament.
    Pairs winners from the completed current round. Assigns BYE if necessary.
    (Standard knockout behavior: BYE goes to leftover player after random pairing)
    """
    bracket_type = match['bracket']
    current_round = match['round']
    winner = match['winner']

    if bracket_type == "Beginner": matches = st.session_state.beginner_matches
    elif bracket_type == "Casual": matches = st.session_state.casual_matches
    else: matches = st.session_state.competitive_matches

    round_matches = [m for m in matches if m['round'] == current_round]
    completed_round_matches = [m for m in round_matches if m['status'] == 'Completed']

    if len(round_matches) == len(completed_round_matches):
        log_event(f"All matches in {bracket_type} Round {current_round} are now complete!")
        next_round = current_round + 1
        current_round_winners = [m['winner'] for m in completed_round_matches if m['winner'] is not None]

        next_round_players_scheduled = {m[p] for m in matches if m['round'] == next_round for p in ['player1', 'player2'] if m[p] != "BYE"}
        available_winners = [w for w in current_round_winners if w not in next_round_players_scheduled]

        if len(available_winners) == 1 and not next_round_players_scheduled:
             log_event(f"🏆 {bracket_type} Tournament Winner: {available_winners[0]} 🏆")
             # Update winner's global status maybe?
             for p in st.session_state.players:
                 if p['name'] == available_winners[0]:
                     p['status'] = 'Tournament Winner'
                     break
             return

        random.shuffle(available_winners)
        match_created_flag = False

        for i in range(0, len(available_winners), 2):
            if i + 1 < len(available_winners):
                player1, player2 = available_winners[i], available_winners[i+1]
                new_match = create_match(player1, player2, bracket_type, next_round)
                matches.append(new_match)
                log_event(f"Created {bracket_type} match: {player1} vs {player2} (Round {next_round})")
                match_created_flag = True
            else:
                bye_player = available_winners[i]
                bye_match = create_match(bye_player, "BYE", bracket_type, next_round)
                bye_match.update({'status': 'Completed', 'winner': bye_player, 'player1_score': 1, 'player2_score': 0})
                matches.append(bye_match)
                log_event(f"{bracket_type} player {bye_player} gets a bye in Round {next_round}")
                update_player_stats(bye_player, 1, True, bracket_type)
                match_created_flag = True
                # Check if this BYE completes the *next* round immediately
                create_next_round_matches(bye_match) # Recursive call to check completion

        if match_created_flag:
             update_rounds_counter(bracket_type, next_round)

    else: # Round not complete, try pairing the current winner if possible
        next_round_players_scheduled = {m[p] for m in matches if m['round'] == current_round + 1 for p in ['player1', 'player2'] if m[p] != "BYE"}
        other_waiting_winners = [m['winner'] for m in completed_round_matches if m['match_id'] != match['match_id'] and m['winner'] != winner and m['winner'] not in next_round_players_scheduled]

        if other_waiting_winners:
            opponent = other_waiting_winners[0]
            new_match = create_match(winner, opponent, bracket_type, current_round + 1)
            matches.append(new_match)
            log_event(f"Created {bracket_type} match: {winner} vs {opponent} (Round {current_round + 1})")
            update_rounds_counter(bracket_type, current_round + 1)
        else:
            log_event(f"{bracket_type} player {winner} is waiting for an opponent in Round {current_round + 1}")


def create_match_for_new_player(player_name, bracket_type):
    """ Attempts to create a match for a newly added player based on tournament format. """
    target_round = 1
    if bracket_type == "Beginner":
        bracket, matches = st.session_state.beginner_bracket, st.session_state.beginner_matches
        if st.session_state.beginner_rounds > 0: target_round = st.session_state.beginner_rounds
    elif bracket_type == "Casual":
        bracket, matches = st.session_state.casual_bracket, st.session_state.casual_matches
        if st.session_state.casual_rounds > 0: target_round = st.session_state.casual_rounds
    else: # Competitive
        bracket, matches = st.session_state.competitive_bracket, st.session_state.competitive_matches
        if st.session_state.competitive_rounds > 0: target_round = st.session_state.competitive_rounds

    player_names_in_bracket = {p['name'] for p in bracket}
    pending_player_names = {m[p] for m in matches if m['status'] == 'Pending' for p in ['player1', 'player2'] if m[p] != "BYE" and m[p] in player_names_in_bracket}
    waiting_players = [p['name'] for p in bracket if p['name'] != player_name and p['name'] not in pending_player_names]

    if waiting_players:
        opponent_name = None
        if st.session_state.tournament_format == "skill_based":
            new_player_skill = get_player_skill(player_name)
            waiting_players.sort(key=lambda p_name: abs(get_player_skill(p_name) - new_player_skill))
            opponent_name = waiting_players[0]
        else: # Standard format
            opponent_name = random.choice(waiting_players)

        new_match = create_match(player_name, opponent_name, bracket_type, target_round)
        matches.append(new_match)
        log_event(f"Created {bracket_type} match for new player: {player_name} vs {opponent_name} (Round {target_round})")
        update_rounds_counter(bracket_type, target_round)
        return True
    else: # No waiting players
        if st.session_state.tournament_format == "standard":
             bye_match = create_match(player_name, "BYE", bracket_type, target_round)
             bye_match.update({'status': 'Completed', 'winner': player_name, 'player1_score': 1, 'player2_score': 0})
             matches.append(bye_match)
             log_event(f"New {bracket_type} player {player_name} gets a BYE (Round {target_round}) as no opponents were available.")
             update_player_stats(player_name, 1, True, bracket_type)
             update_rounds_counter(bracket_type, target_round)
             create_next_round_matches(bye_match) # Check if this completes round
             return True
        else: # Skill-based format: Player waits
             log_event(f"New {bracket_type} player {player_name} is waiting for an opponent (Skill-Based Mode).")
             return False


def handle_match_result(match, player1_score, player2_score):
    """ Processes the result of a completed match, updates stats, and triggers next round logic. """
    try: p1s, p2s = int(player1_score), int(player2_score)
    except (ValueError, TypeError): return False, "Scores must be valid numbers."

    if match['player1'] == "BYE" or match['player2'] == "BYE": return True, None # Already handled

    if p1s == p2s: return False, "Scores cannot be tied."

    winner, loser = (match['player1'], match['player2']) if p1s > p2s else (match['player2'], match['player1'])
    winner_score, loser_score = (p1s, p2s) if p1s > p2s else (p2s, p1s)

    match.update({'player1_score': p1s, 'player2_score': p2s, 'winner': winner, 'status': 'Completed'})
    bracket_type = match['bracket']

    update_player_stats(match['player1'], p1s, match['player1'] == winner, bracket_type)
    update_player_stats(match['player2'], p2s, match['player2'] == winner, bracket_type)
    log_event(f"{bracket_type} match result (R{match['round']}): {winner} ({winner_score}) defeated {loser} ({loser_score})")

    if st.session_state.tournament_format == "standard":
        create_next_round_matches(match)
    # No automatic progression needed here for skill-based

    return True, None


# --- Visualization Functions ---

def create_bracket_visualization(bracket_type):
    """ Generates a Plotly figure visualizing the tournament bracket. """
    if bracket_type == "Beginner": matches = st.session_state.beginner_matches
    elif bracket_type == "Casual": matches = st.session_state.casual_matches
    else: matches = st.session_state.competitive_matches

    fig = go.Figure()
    fig.update_layout(title=f"{bracket_type} Bracket Visualization", showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=10, r=10, t=40, b=10))

    if not matches:
        fig.add_annotation(text=f"No matches in {bracket_type} bracket yet", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        return fig

    rounds_data = {}
    all_round_nums = set(m['round'] for m in matches)
    max_round_num = max(all_round_nums) if all_round_nums else 0
    for r in range(1, max_round_num + 2): rounds_data[r] = sorted([m for m in matches if m['round'] == r], key=lambda x: x['created_at']) # Sort matches within round

    node_positions = {}
    max_matches_in_round = max((len(rounds_data[r]) for r in rounds_data if rounds_data[r]), default=1)
    plot_height = max(600, 120 * max_matches_in_round)
    plot_width = (max_round_num + 1) * 200
    fig.update_layout(height=plot_height)

    for round_num in sorted(rounds_data.keys()):
        round_matches = rounds_data[round_num]
        if not round_matches: continue

        x_pos = round_num * 180
        num_items = len(round_matches)
        y_spacing = plot_height / (num_items + 1)
        match_y_centers = [(i + 1) * y_spacing for i in range(num_items)]
        processed_players_in_round = set()

        for i, match in enumerate(round_matches):
            match_center_y = match_y_centers[i]
            player1, player2 = match['player1'], match['player2']
            is_bye = (player1 == "BYE" or player2 == "BYE")
            status = match['status']; winner = match['winner']
            players_in_match = [p for p in [player1, player2] if p != "BYE"]
            y_offset = 30

            for j, player in enumerate(players_in_match):
                 player_y = match_center_y if is_bye else (match_center_y - y_offset if j == 0 else match_center_y + y_offset)
                 node_key = (round_num, player)
                 # If player already positioned in this round (e.g., from a BYE), reuse Y slightly offset
                 if node_key in node_positions:
                      player_y = node_positions[node_key][1] + (5 if j == 1 else -5) # Adjust slightly if collision

                 node_positions[node_key] = (x_pos, player_y)

                 if node_key in processed_players_in_round: continue
                 processed_players_in_round.add(node_key)

                 node_color, node_text, hover_text = 'grey', player, f"Player: {player}<br>Round: {round_num}"
                 if is_bye: node_color, node_text, hover_text = 'orange', f"{player} (BYE)", f"{player} received a BYE in Round {round_num}"
                 elif status == 'Pending': node_color = 'blue'
                 elif status == 'Completed':
                     is_winner = (player == winner)
                     score = match['player1_score'] if player == player1 else match['player2_score']
                     opponent = player2 if player == player1 else player1
                     node_color = 'green' if is_winner else 'red'
                     node_text = f"{player} ({score})"
                     hover_text = f"{'Win' if is_winner else 'Loss'}: {player} ({score})<br>vs {opponent}<br>Round: {round_num}"

                 fig.add_trace(go.Scatter(x=[x_pos], y=[player_y], mode='markers+text', marker=dict(size=12, color=node_color), text=[node_text], textposition="middle right", textfont=dict(size=10), hoverinfo='text', hovertext=hover_text, name=player))

            if round_num > 1:
                 prev_round_num = round_num - 1
                 for player in players_in_match:
                     origin_pos = node_positions.get((prev_round_num, player))
                     current_pos = node_positions.get((round_num, player))
                     if origin_pos and current_pos:
                         line_color = 'lightgrey'
                         prev_match = next((m for m in rounds_data.get(prev_round_num, []) if player in [m['player1'], m['player2']]), None)
                         if prev_match and prev_match['status'] == 'Completed': line_color = 'green' if prev_match['winner'] == player else 'red'
                         fig.add_trace(go.Scatter(x=[origin_pos[0], current_pos[0]], y=[origin_pos[1], current_pos[1]], mode='lines', line=dict(color=line_color, width=1.5), hoverinfo='none'))

    for r in range(1, max_round_num + 1):
         fig.add_annotation(x=r * 180, y=plot_height - 15, text=f"<b>Round {r}</b>", showarrow=False, font=dict(size=14), xanchor="center")

    fig.update_xaxes(range=[80, plot_width + 80])
    fig.update_yaxes(range=[0, plot_height])
    return fig


# --- UI Rendering Functions ---

def render_match_ui(match, bracket_type):
    """ Renders the UI elements for inputting scores for a pending match. """
    if match['player1'] == "BYE" or match['player2'] == "BYE" or match['status'] == 'Completed': return

    container = st.container(border=True)
    container.markdown(f"###### Match {match['match_id']} (R{match['round']})")
    container.markdown(f"**{match['player1']}** vs **{match['player2']}**")
    col1, col2 = container.columns(2)
    with col1: player1_score = st.number_input(f"{match['player1']} Score:", min_value=0, value=None, key=f"p1_score_{bracket_type.lower()}_{match['match_id']}", placeholder="Score")
    with col2: player2_score = st.number_input(f"{match['player2']} Score:", min_value=0, value=None, key=f"p2_score_{bracket_type.lower()}_{match['match_id']}", placeholder="Score")

    if container.button("Record Result", key=f"btn_{bracket_type.lower()}_{match['match_id']}", use_container_width=True):
        if player1_score is None or player2_score is None: container.warning("Enter scores for both players.")
        else:
             success, error_msg = handle_match_result(match, player1_score, player2_score)
             if not success: container.error(error_msg)
             else: st.success(f"Result recorded for {match['match_id']}!"); st.rerun()


def render_bracket_tab(bracket_type):
    """ Renders the content for a specific bracket tab. """
    st.header(f"{bracket_type} Bracket")
    st.plotly_chart(create_bracket_visualization(bracket_type), use_container_width=True)

    st.subheader("Pending Matches")
    if bracket_type == "Beginner": matches, current_round_num = st.session_state.beginner_matches, st.session_state.beginner_rounds
    elif bracket_type == "Casual": matches, current_round_num = st.session_state.casual_matches, st.session_state.casual_rounds
    else: matches, current_round_num = st.session_state.competitive_matches, st.session_state.competitive_rounds

    pending_matches = [m for m in matches if m['status'] == 'Pending' and m['player1'] != "BYE" and m['player2'] != "BYE"]

    if not pending_matches:
        st.info("No pending matches.")
        if st.session_state.tournament_started and st.session_state.tournament_format == "skill_based":
            bracket_list = getattr(st.session_state, f"{bracket_type.lower()}_bracket", [])
            player_names_in_bracket = {p['name'] for p in bracket_list}
            pending_player_names = {m[p] for m in matches if m['status'] == 'Pending' for p in ['player1', 'player2'] if m[p] != "BYE" and m[p] in player_names_in_bracket}
            available_count = len(player_names_in_bracket) - len(pending_player_names)
            if available_count >= 1:
                 if st.button(f"Create New Round {current_round_num + 1} Matches", key=f"new_{bracket_type.lower()}_matches"):
                     if create_skill_based_matches(bracket_type): st.success(f"New matches/BYEs created for Round {current_round_num + 1}."); st.rerun()
    else:
        lowest_pending_round = min(m['round'] for m in pending_matches)
        matches_to_display = sorted([m for m in pending_matches if m['round'] == lowest_pending_round], key=lambda m: m['created_at'])
        st.markdown(f"**Round {lowest_pending_round}**")
        cols = st.columns(min(3, len(matches_to_display)))
        for i, match in enumerate(matches_to_display):
            with cols[i % 3]: render_match_ui(match, bracket_type)

    st.subheader("Completed Matches")
    completed_matches = [m for m in matches if m['status'] == 'Completed']
    if completed_matches:
        try:
            match_df = pd.DataFrame(completed_matches)
            display_cols = ['round', 'player1', 'player1_score', 'player2', 'player2_score', 'winner', 'match_id', 'created_at']
            match_df = match_df[display_cols].sort_values(by=['round', 'created_at'], ascending=[True, True])
            st.dataframe(match_df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Error displaying completed matches: {e}")
            st.dataframe(pd.DataFrame(completed_matches), hide_index=True)
    else: st.info("No completed matches yet.")


def render_players_stats(bracket_type):
    """ Renders player statistics for a specific bracket with updated formatting. """
    st.header(f"{bracket_type} Player Stats")
    bracket_list = getattr(st.session_state, f"{bracket_type.lower()}_bracket", [])
    players_in_bracket_names = {p['name'] for p in bracket_list}
    bracket_players_data = [p for p in st.session_state.players if p['name'] in players_in_bracket_names]

    if bracket_players_data:
        df = pd.DataFrame(bracket_players_data)
        for col in ['wins', 'losses', 'total_score', 'avg_score']: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df['avg_score'] = df['avg_score'].round(0).astype(int)
        display_df = df[['name', 'wins', 'losses', 'total_score', 'avg_score']].sort_values(by=['wins', 'avg_score'], ascending=[False, False])
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        if not display_df.empty:
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(x=display_df['name'], y=display_df['avg_score'], text=display_df['avg_score'].apply(lambda x: f'{x:.0f}'), textposition='auto', marker_color='#1f77b4'))
            fig_bar.update_layout(title=f'{bracket_type} Bracket - Average Scores per Match', xaxis_title='Player', yaxis_title='Average Score (Rounded)', height=400, xaxis={'categoryorder':'total descending'})
            st.plotly_chart(fig_bar, use_container_width=True)
    else: st.info(f"No players have participated in the {bracket_type.lower()} bracket yet.")


def render_top_scores():
    """ Renders overall top match scores across all brackets. """
    st.header("Overall Top Match Scores")
    all_matches = st.session_state.beginner_matches + st.session_state.casual_matches + st.session_state.competitive_matches
    completed_matches = [m for m in all_matches if m['status'] == 'Completed' and m['player1'] != 'BYE' and m['player2'] != 'BYE']

    if completed_matches:
        scores_data = [{'player': m[p], 'score': m[f'{p}_score'], 'bracket': m['bracket'], 'round': m['round']} for m in completed_matches for p in ['player1', 'player2']]
        if scores_data:
            top_df = pd.DataFrame(scores_data)
            top_df['score'] = pd.to_numeric(top_df['score'], errors='coerce').fillna(0)
            top_df = top_df.sort_values('score', ascending=False).head(10)

            fig_top = go.Figure()
            colors = ['#FFD700', '#C0C0C0', '#CD7F32'] + ['#1f77b4'] * 7
            fig_top.add_trace(go.Bar(x=top_df['player'], y=top_df['score'], text=top_df['score'].apply(lambda x: f'{x:,.0f}'), textposition='auto', marker_color=colors[:len(top_df)], name='Top Scores', hovertemplate='<b>%{x}</b><br>Score: %{y:,}<br>Bracket: %{customdata[0]}<br>Round: %{customdata[1]}<extra></extra>', customdata=top_df[['bracket', 'round']].values))
            fig_top.update_layout(title='Top 10 Individual Match Scores (All Brackets)', xaxis_title='Player', yaxis_title='Score', height=450, xaxis={'categoryorder':'total descending'})
            st.plotly_chart(fig_top, use_container_width=True)
        else: st.info("No completed match scores available yet.")
    else: st.info("No completed matches with scores yet.")


# --- Main Application ---
st.title("🎸 Guitar Hero Tournament Manager 🎸")

# --- Sidebar for Setup and Controls ---
with st.sidebar:
    st.header("Tournament Setup")
    # Only show setup controls if tournament hasn't started
    if not st.session_state.tournament_started:
        st.session_state.tournament_format = st.radio("Select Tournament Format:", ("standard", "skill_based"), index=("standard", "skill_based").index(st.session_state.tournament_format), format_func=lambda x: "Standard Knockout" if x == "standard" else "Skill-Based Pairing", key='tournament_format_radio', help="Standard: Random pairing knockout. Skill-Based: Pairs players by skill, random BYE for odd player (fairness attempted).")
        st.markdown(f"**Format:** `{st.session_state.tournament_format}`")

        st.subheader("Add Player")
        with st.form("add_player_form", clear_on_submit=True):
            new_player_name = st.text_input("Player Name:")
            player_skill_level = st.selectbox("Skill Level:", ("Beginner", "Casual", "Competitive"))
            submitted = st.form_submit_button("Add Player")
            if submitted:
                if new_player_name:
                    if any(p['name'] == new_player_name for p in st.session_state.players): st.warning(f"Player '{new_player_name}' already exists.")
                    else:
                        st.session_state.players.append({'name': new_player_name, 'skill_level': player_skill_level, 'wins': 0, 'losses': 0, 'total_score': 0, 'avg_score': 0, 'status': 'Active'})
                        bracket_list = getattr(st.session_state, f"{player_skill_level.lower()}_bracket")
                        bracket_list.append({'name': new_player_name, 'status': 'Active'})
                        log_event(f"Player '{new_player_name}' added to {player_skill_level} bracket.")
                        st.success(f"Player '{new_player_name}' added!"); st.rerun()
                else: st.warning("Please enter a player name.")

        if st.session_state.players and not st.session_state.tournament_started:
             if st.button("Start Tournament", type="primary", use_container_width=True):
                 st.session_state.tournament_started = True
                 log_event(f"Tournament started with {st.session_state.tournament_format} format.")
                 # Create initial matches
                 for bracket_type in ["Beginner", "Casual", "Competitive"]:
                     bracket_list = getattr(st.session_state, f"{bracket_type.lower()}_bracket")
                     if len(bracket_list) >= 1:
                         if st.session_state.tournament_format == "standard":
                             players_in_bracket = [p['name'] for p in bracket_list]; random.shuffle(players_in_bracket)
                             matches_list = getattr(st.session_state, f"{bracket_type.lower()}_matches")
                             for i in range(0, len(players_in_bracket), 2):
                                 if i + 1 < len(players_in_bracket):
                                     p1, p2 = players_in_bracket[i], players_in_bracket[i+1]
                                     match = create_match(p1, p2, bracket_type, 1); matches_list.append(match)
                                     log_event(f"Created initial {bracket_type} match: {p1} vs {p2} (R1)")
                                 else:
                                     bye_player = players_in_bracket[i]
                                     bye_match = create_match(bye_player, "BYE", bracket_type, 1)
                                     bye_match.update({'status': 'Completed', 'winner': bye_player, 'player1_score': 1, 'player2_score': 0})
                                     matches_list.append(bye_match); update_player_stats(bye_player, 1, True, bracket_type)
                                     log_event(f"Initial {bracket_type} BYE for {bye_player} (R1)")
                             if players_in_bracket: update_rounds_counter(bracket_type, 1)
                         elif st.session_state.tournament_format == "skill_based":
                              create_skill_based_matches(bracket_type)
                 st.rerun()
    else: # Tournament started
        st.info(f"Tournament in progress ({st.session_state.tournament_format} format).")
        st.subheader("Add Late Player")
        with st.form("add_late_player_form", clear_on_submit=True):
            late_player_name = st.text_input("Late Player Name:")
            late_player_skill = st.selectbox("Skill Level:", ("Beginner", "Casual", "Competitive"))
            submitted_late = st.form_submit_button("Add Late Player")
            if submitted_late:
                 if late_player_name:
                     if any(p['name'] == late_player_name for p in st.session_state.players): st.warning(f"Player '{late_player_name}' already exists.")
                     else:
                         st.session_state.players.append({'name': late_player_name, 'skill_level': late_player_skill, 'wins': 0, 'losses': 0, 'total_score': 0, 'avg_score': 0, 'status': 'Active'})
                         bracket_list = getattr(st.session_state, f"{late_player_skill.lower()}_bracket")
                         bracket_list.append({'name': late_player_name, 'status': 'Active'})
                         log_event(f"Late player '{late_player_name}' added to {late_player_skill} bracket.")
                         st.success(f"Late player '{late_player_name}' added.")
                         create_match_for_new_player(late_player_name, late_player_skill)
                         st.rerun()
                 else: st.warning("Please enter a player name.")

# --- Main Area ---
if not st.session_state.tournament_started:
    tab1, tab2 = st.tabs(["Players", "Log"])
    with tab1:
        st.header("Registered Players")
        if st.session_state.players:
            player_df = pd.DataFrame(st.session_state.players)
            st.metric("Total Players", len(st.session_state.players))
            st.dataframe(player_df[['name', 'skill_level']], use_container_width=True, hide_index=True)
        else: st.info("No players added yet. Use the sidebar to add players.")
    with tab2:
        st.header("Log")
        # FIX: Use a standard loop instead of list comprehension for log display
        log_container = st.container(height=400)
        for entry in st.session_state.tournament_log:
            log_container.text(entry) # Display each entry
else:
    tab_titles = ["Beginner", "Casual", "Competitive", "Overall Stats", "Top Scores", "Log"]
    tabs = st.tabs(tab_titles)
    with tabs[0]: render_bracket_tab("Beginner"); render_players_stats("Beginner")
    with tabs[1]: render_bracket_tab("Casual"); render_players_stats("Casual")
    with tabs[2]: render_bracket_tab("Competitive"); render_players_stats("Competitive")
    with tabs[3]: # Overall Stats
        st.header("Overall Player Statistics")
        if st.session_state.players:
             overall_df = pd.DataFrame(st.session_state.players)
             for col in ['wins', 'losses', 'total_score', 'avg_score']: overall_df[col] = pd.to_numeric(overall_df[col], errors='coerce').fillna(0)
             overall_df['avg_score'] = overall_df['avg_score'].round(0).astype(int) # Format avg_score
             st.dataframe(overall_df[['name', 'skill_level', 'wins', 'losses', 'avg_score']], use_container_width=True, hide_index=True)
        else: st.info("No players in the tournament.")
    with tabs[4]: render_top_scores()
    with tabs[5]: # Log
        st.header("Tournament Log")
        # FIX: Use a standard loop instead of list comprehension for log display
        log_container = st.container(height=500)
        for entry in st.session_state.tournament_log:
            log_container.text(entry) # Display each entry

