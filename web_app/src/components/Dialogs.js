import * as React from "react"
import Button from "@mui/material/Button";
import Dialog from "@mui/material/Dialog"
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogContentText from "@mui/material/DialogContentText";
import DialogTitle from "@mui/material/DialogTitle"
import Paper from "@mui/material/Paper";
import Draggable from "react-draggable";

function PaperComponent(props) {
    return (
        <Draggable
        handle="#draggable-dialog-title"
        cancel={'[class*="MuiDialogContent-root"]'}
        >
        <Paper {...props} />
        </Draggable>
    );
}

export function DraggableDialog (props) {
    const [open, setOpen] = React.useState(false);

    const handleClickOpen = () => {
        setOpen(true);
        return props.OkEvent;
    }
    const handleClose = () => {
        setOpen(false);
        return props.CancelEvent;
    }
    return (
        <React.Fragment>
            <Button variant="outlined" onClick={handleClickOpen}></Button>
            <Dialog
            open={open}
            onClose={handleClose}
            PaperComponent={PaperComponent}
            aria-labelledby="draggable-dialog-title"
            >
                <DialogTitle style={{ cursor: 'move' }} id="draggable-dialog-title">
                    {props.Title}
                </DialogTitle>

                <DialogContent>
                    <DialogContentText>
                        {props.Content}
                    </DialogContentText>
                </DialogContent>

                <DialogActions>
                    <Button autoFocus onClick={handleClose}>
                        {props.TextCancel}
                    </Button>
                    <Button autoFocus onClick={handleClickOpen}>
                        {props.TextOK}
                    </Button>
                </DialogActions>
                
            </Dialog>
        </React.Fragment>
    );
}