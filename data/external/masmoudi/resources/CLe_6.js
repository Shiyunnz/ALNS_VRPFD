WS.Namespace.Create('NovAxis.ToggleButton', (function ()
{
    /**
    @class ToggleButton
    @namespace NovAxis
    @constructor
    @param {JQuery Dom} Element Dom element that will be replace by the control. <br/>
    Must be int the specific form : &lt;div&gt;&lt;span name="on"&gt;On text&lt;/span&gt;&lt;span name="off"&gt;Off text&lt;/span&gt;&lt;/div&gt;<br/>
    Class "on" is supported to set button value : &lt;div class="on"&gt;&lt;span name="on"&gt;On text&lt;/span&gt;&lt;span name="off"&gt;Off text&lt;/span&gt;&lt;/div&gt;<br/>
    @param {Boolean} [Value] true, false
    @param {Array<string>} [Classes] Array of class name to apply to the control.
    **/
    var _class = function (Element, Value, Classes)
    {
        if (!Element || Element.length == 0) throw "Element control cannot be null.";
        if (Value && (typeof Value !== "boolean")) throw "Value must be boolean.";


        /**
        Fired when the value is changed.

        @event change
        @param {Boolean} The new value.
        **/
        WS.EventObject.call(this, ['change']);

        // InitData
        this._Value = false;
        this.originalElement = Element;

        // Element
        this._Element = $(CreateControl.call(this, Element));
        
        Element.after(this._Element);
        Element.remove();
        if (Classes && Classes.length > 0) this._Element.addClass(Classes.join(" "));

        // Events, Bindings
        AddEvents.apply(this);
        if (Value != null) this.SetValue(Value);
    };

    WS.Exts.Inherits(_class, WS.EventObject);

    // Private Method(s)
    var CreateControl = function (Element)
    {
        var Html = [];

        var MyElement = {
            Button: Element,
            On: Element.find('span[name=on]'),
            Off: Element.find('span[name=off]')
        };

        var On = "";

        if (MyElement.Button.hasClass('on'))
        {
            On = "class=\"wsOn\"";
            this._Value = true;
        }
        else
        {           
            this._Value = false;
        }

        Html.push("<div class=\"naToggleButton\">");
        Html.push("<div " + On + ">");
        Html.push(MyElement.On[0].outerHTML);
        Html.push(MyElement.Off[0].outerHTML);
        Html.push("<div></div>");
        Html.push("</div>");
        Html.push("</div>");

        return Html.join("");
    };

    var AddEvents = function ()
    {
        var Me = this;
        var button = this._Element.find("> div");

        button.off('click').on('click', function ()
        {
            var Value = !Me._Value;
            Me.SetValue(Value);
            Me.RaiseEvent('change', Value);
        });
    };

    // Public Method(s)
    (function (Methods)
    {
        /**
        Get the selected value (on/off).
    
        @method GetValue
        @return {boolean} Value
        **/
        Methods.GetValue = function ()
        {
            return this._Value;
        };

        /**
        Set the selected value (on/off).
    
        @method SetValue
        @param {boolean} Value
        **/
        Methods.SetValue = function (Value)
        {
            if (Value == null) throw 'Value cannot be null.';
            if (Value && (typeof Value !== 'boolean')) throw 'Value must be boolean.';

            this._Value = Value;
            var button = this._Element.find('> div');
            button.removeClass('wsOn');
            if (this._Value) button.addClass('wsOn');
        };

        /**
        Disable the control.
   
        @method Disable
        **/
        Methods.Disable = function ()
        {
            this._Element.find('> div').css({ 'pointer-events': 'none', 'background-color': '#5c5c5c' });
        };

        /**
        Enable the control.
 
        @method Enable
        **/
        Methods.Enable = function ()
        {
            this._Element.find('> div').css({ 'pointer-events': 'all', 'background-color': '' });
        };

        /**
        Destroy the control.
 
        @method Destroy
        **/
        Methods.Destroy = function ()
        {
            this._Element.after(this.originalElement);
            this._Element.remove();
            this._Element = null;
        };

    })(_class.prototype);

    return _class;
})());