from flask import Blueprint, request, jsonify

fetch_gpt_response_bp = Blueprint('fetch_gpt_response', __name__)

from backend.ai_agent import GPTDependencies, create_agent, get_recommendations, HISTORICAL_DATA

agent = create_agent()

@fetch_gpt_response_bp.route('/api/gpt_response', methods=['POST'])
async def fetch_gpt_response():
    try:
        data = request.get_json()
        context = data.get('field_context')
        grid_raw  = data.get('gridData')
        area_raw  = data.get('areaData')

        deps = GPTDependencies(
            context=context,
            gridData=grid_raw,
            areaData=area_raw,         
            historical_context=HISTORICAL_DATA
        )

        gpt_out = await get_recommendations(agent, deps)

        print("")
        print("")
        print("------GPT OUT:-----")
        print(gpt_out)
        print("")
        print("")

        return gpt_out.tips

    except Exception as e:
        return jsonify({"error": str(e)}), 500