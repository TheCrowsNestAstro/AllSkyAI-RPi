from waitress import serve
import allskyai_app
serve(allskyai_app.app, host='0.0.0.0', port=3010)
