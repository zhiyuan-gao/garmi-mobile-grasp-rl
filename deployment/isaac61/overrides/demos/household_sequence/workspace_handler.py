"""Allow the packaged Workspace to use its authenticated same-origin Lab panes."""
from jupyter_server.files.handlers import FilesHandler

class WorkspaceFilesHandler(FilesHandler):
    @property
    def content_security_policy(self):
        policy=super().content_security_policy
        # Keep Jupyter's normal HTML sandbox for every other uploaded file.
        if self.request.path.endswith('/files/demos/web_ui/index.html'):
            return policy.replace('; sandbox allow-scripts', '; sandbox allow-scripts allow-same-origin allow-forms allow-popups')
        return policy
