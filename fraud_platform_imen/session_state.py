class SessionState:
    def __init__(self, current_page="login", user_logged_in=False):
        self.current_page = current_page
        self.user_logged_in = user_logged_in


    def get_current_page(self, default=None):
        return self.current_page

    def get_user_logged_in(self, default=None):
        return self.user_logged_in

    def set_current_page(self, page, default=None):
        self.current_page = page


    def set_user_logged_in(self, value, default=None):
        self.user_logged_in = value
