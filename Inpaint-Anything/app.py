from flask import Flask, request, jsonify
from ai_engine import sota_inpaint_engine, sota_move_engine

app = Flask(__name__)

@app.route('/api/image-gen', methods=['POST'])
def inpaint_endpoint():
    data = request.json
    intent = data.get('intent', 'edit')  
    image_path = data.get('image_path')

    try:
        if intent == "move":
            source_bbox = data.get('source_bbox')
            dest_bbox = data.get('dest_bbox')
            position = data.get('position', 'center') # <--- ΔΙΑΒΑΖΕΙ ΤΗ ΘΕΣΗ
            
            if not source_bbox or not dest_bbox:
                return jsonify({"status": "error", "message": "Missing coordinates"}), 400
                
            # Την περνάει στη συνάρτηση
            result_path = sota_move_engine(image_path, source_bbox, dest_bbox, position)
            
        else:
            bboxes = data.get('bboxes', [])
            result_path = sota_inpaint_engine(image_path, bboxes)

        return jsonify({"status": "success", "response": result_path})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4444)
