from shiny import App

from server.core import server
from ui.layout import create_app_ui

app = App(create_app_ui(), server)
